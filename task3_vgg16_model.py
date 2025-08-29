class VGG16EdgeDetection(nn.Module):
    def __init__(self):
        super(VGG16EdgeDetection, self).__init__()
        
        # Load pretrained VGG16 model without classifier
        vgg16 = models.vgg16(pretrained=True)
        
        # Extract features (without the last max pooling layer)
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])
        
        # Transpose convolution decoder to restore original image size
        # Input: 512 channels from VGG16 (after dropping last pooling layer)
        # Size is 1/16 of original (due to 4 pooling operations)
        self.decoder = nn.Sequential(
            # First upsampling: 1/16 -> 1/8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Second upsampling: 1/8 -> 1/4
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Third upsampling: 1/4 -> 1/2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth upsampling: 1/2 -> 1
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            
            # Final convolution to get single channel
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Get original dimensions for later upsampling
        _, _, h, w = x.size()
        
        # Extract features using VGG16
        x = self.features(x)
        
        # Decode using transpose convolution
        x = self.decoder(x)
        
        return x

class VGG16BilinearEdgeDetection(nn.Module):
    def __init__(self):
        super(VGG16BilinearEdgeDetection, self).__init__()
        
        # Load pretrained VGG16 model without classifier
        vgg16 = models.vgg16(pretrained=True)
        
        # Extract features (without the last max pooling layer)
        self.features = nn.Sequential(*list(vgg16.features.children())[:-1])
        
        # Simple decoder with bilinear upsampling
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        # Get original dimensions for later upsampling
        _, _, h, w = x.size()
        
        # Extract features using VGG16
        x = self.features(x)
        
        # Apply convolutions
        x = self.decoder(x)
        
        # Bilinear upsampling to match original image size
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return x

def task3_vgg16_model():
    """
    Task 3: Train VGG16-based models for edge detection
    """
    print("Task 3: VGG16 Model")
    
    # Set up transforms for loading images
    transform = transforms.Compose([
        transforms.Resize((320, 480)),  # Resize to make it more manageable
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = BSDS500Dataset(TRAIN_IMG_DIR, TRAIN_GT_DIR, transform=transform)
    val_dataset = BSDS500Dataset(VAL_IMG_DIR, VAL_GT_DIR, transform=transform)
    test_dataset = BSDS500Dataset(TEST_IMG_DIR, TEST_GT_DIR, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
    
    # Initialize models
    vgg_transconv_model = VGG16EdgeDetection().to(device)
    vgg_bilinear_model = VGG16BilinearEdgeDetection().to(device)
    
    # Loss function - using the class balanced cross entropy loss
    criterion = BalancedCrossEntropyLoss()
    
    # Optimizer
    optimizer_transconv = optim.Adam(vgg_transconv_model.parameters(), lr=0.0001)
    optimizer_bilinear = optim.Adam(vgg_bilinear_model.parameters(), lr=0.0001)
    
    # Train the models
    print("Training VGG16 with Transpose Convolution...")
    train_losses_transconv, val_losses_transconv = train_model(
        vgg_transconv_model, train_loader, val_loader, criterion, optimizer_transconv, 
        num_epochs=50, model_name="vgg16_transconv"
    )
    
    print("Training VGG16 with Bilinear Upsampling...")
    train_losses_bilinear, val_losses_bilinear = train_model(
        vgg_bilinear_model, train_loader, val_loader, criterion, optimizer_bilinear, 
        num_epochs=50, model_name="vgg16_bilinear"
    )
    
    # Compare loss curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses_transconv)+1), train_losses_transconv, label='Transpose Conv')
    plt.plot(range(1, len(train_losses_bilinear)+1), train_losses_bilinear, label='Bilinear')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_losses_transconv)+1), val_losses_transconv, label='Transpose Conv')
    plt.plot(range(1, len(val_losses_bilinear)+1), val_losses_bilinear, label='Bilinear')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("vgg16_loss_comparison.png")
    plt.show()
    
    # Evaluate on test set
    print("Evaluating VGG16 with Transpose Convolution...")
    evaluate_model(vgg_transconv_model, test_loader, threshold=0.5, model_name="VGG16_TransConv")
    
    print("Evaluating VGG16 with Bilinear Upsampling...")
    evaluate_model(vgg_bilinear_model, test_loader, threshold=0.5, model_name="VGG16_Bilinear")
    
    print("Task 3 completed.")
    print("Discussion on VGG16 model:")
    print("1. VGG16 uses more complex features from pretrained ImageNet model")
    print("2. Transpose convolution vs bilinear upsampling comparison:")
    print("   - Transpose convolution is learnable and can capture more details")
    print("   - Bilinear upsampling is simpler but may produce smoother edges")
    print("3. The class-balanced loss function is still crucial due to edge/non-edge imbalance")
    print("4. VGG16 should outperform the simple CNN due to deeper architecture and pretraining")

if __name__ == "__main__":
    task3_vgg16_model()
