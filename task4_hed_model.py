class HEDModel(nn.Module):
    def __init__(self):
        super(HEDModel, self).__init__()
        
        # Load pretrained VGG16 features
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        
        # Extract VGG layers before each pooling
        # VGG16 structure: 2 convs -> pool -> 2 convs -> pool -> 3 convs -> pool -> 3 convs -> pool -> 3 convs
        
        # First stage: conv1_1, conv1_2
        self.stage1 = nn.Sequential(*features[:4])
        
        # Second stage: conv2_1, conv2_2
        self.stage2 = nn.Sequential(*features[4:9])
        
        # Third stage: conv3_1, conv3_2, conv3_3
        self.stage3 = nn.Sequential(*features[9:16])
        
        # Fourth stage: conv4_1, conv4_2, conv4_3
        self.stage4 = nn.Sequential(*features[16:23])
        
        # Fifth stage: conv5_1, conv5_2, conv5_3 (without the final max pooling)
        self.stage5 = nn.Sequential(*features[23:30])
        
        # Side output layers
        self.side1 = nn.Conv2d(64, 1, kernel_size=1)
        self.side2 = nn.Conv2d(128, 1, kernel_size=1)
        self.side3 = nn.Conv2d(256, 1, kernel_size=1)
        self.side4 = nn.Conv2d(512, 1, kernel_size=1)
        self.side5 = nn.Conv2d(512, 1, kernel_size=1)
        
        # Fusion layer
        self.fuse = nn.Conv2d(5, 1, kernel_size=1)
        
        # Initialize weights for side outputs and fusion
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in [self.side1, self.side2, self.side3, self.side4, self.side5, self.fuse]:
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Store original dimensions
        _, _, h, w = x.size()
        
        # Forward through VGG stages
        stage1_out = self.stage1(x)
        stage2_out = self.stage2(stage1_out)
        stage3_out = self.stage3(stage2_out)
        stage4_out = self.stage4(stage3_out)
        stage5_out = self.stage5(stage4_out)
        
        # Side outputs
        side1 = self.side1(stage1_out)
        side2 = self.side2(stage2_out)
        side3 = self.side3(stage3_out)
        side4 = self.side4(stage4_out)
        side5 = self.side5(stage5_out)
        
        # Upsample side outputs to match original image size
        side1 = nn.functional.interpolate(side1, size=(h, w), mode='bilinear', align_corners=False)
        side2 = nn.functional.interpolate(side2, size=(h, w), mode='bilinear', align_corners=False)
        side3 = nn.functional.interpolate(side3, size=(h, w), mode='bilinear', align_corners=False)
        side4 = nn.functional.interpolate(side4, size=(h, w), mode='bilinear', align_corners=False)
        side5 = nn.functional.interpolate(side5, size=(h, w), mode='bilinear', align_corners=False)
        
        # Concatenate all side outputs
        fuse_input = torch.cat((side1, side2, side3, side4, side5), dim=1)
        
        # Fuse side outputs
        fuse = self.fuse(fuse_input)
        
        # Group all outputs for loss computation
        outputs = [side1, side2, side3, side4, side5, fuse]
        
        return outputs

class HEDLoss(nn.Module):
    """
    Loss function for HED as described in the paper
    Computes loss for each side output and the fused output
    """
    def __init__(self):
        super(HEDLoss, self).__init__()
        self.balanced_ce = BalancedCrossEntropyLoss()
    
    def forward(self, outputs, target):
        """
        Args:
            outputs: List of 6 tensors [side1, side2, side3, side4, side5, fuse]
            target: Ground truth edge map
        """
        losses = []
        
        # Compute loss for each side output and the fused output
        for output in outputs:
            losses.append(self.balanced_ce(output, target))
        
        # Total loss is the sum of all losses
        total_loss = sum(losses)
        
        return total_loss, losses

def train_hed_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train the HED model and validate
    """
    train_losses = []
    val_losses = []
    side_losses = [[] for _ in range(6)]  # To track individual side output losses
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_side_losses = [0.0 for _ in range(6)]
        
        # Use tqdm for progress bar
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for images, targets, _ in pbar:
                images = images.to(device)
                targets = targets.to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss, individual_losses = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                for i, ind_loss in enumerate(individual_losses):
                    running_side_losses[i] += ind_loss.item()
                
                pbar.set_postfix({'loss': loss.item()})
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        for i in range(6):
            side_loss = running_side_losses[i] / len(train_loader)
            side_losses[i].append(side_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss, _ = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "hed_model.pth")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss - HED Model')
    plt.legend()
    plt.grid(True)
    plt.savefig("hed_model_loss.png")
    plt.show()
    
    # Plot side output losses
    plt.figure(figsize=(12, 6))
    for i in range(6):
        label = f"Side {i+1}" if i < 5 else "Fused"
        plt.plot(range(1, num_epochs+1), side_losses[i], label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Side Output Losses - HED Model')
    plt.legend()
    plt.grid(True)
    plt.savefig("hed_side_output_losses.png")
    plt.show()
    
    return train_losses, val_losses, side_losses

def evaluate_hed_model(model, test_loader, threshold=0.5):
    """
    Evaluate the HED model on test data and visualize results
    """
    model.eval()
    
    # Get fusion layer weights for visualization
    fusion_weights = model.fuse.weight.data.cpu().numpy().squeeze()
    
    # Print the learned weights
    print("Learned fusion weights:")
    for i, weight in enumerate(fusion_weights):
        if i < 5:  # Individual weights for each side output
            print(f"Side {i+1}: {weight.mean():.4f}")
    
    # Process test samples
    with torch.no_grad():
        for i, (images, targets, img_names) in enumerate(test_loader):
            if i >= 5:  # Limit to 5 test images
                break
                
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get side outputs and fused output
            side_outputs = outputs[:5]
            fused_output = outputs[5]
            
            # Apply sigmoid to get probabilities
            side_probs = [torch.sigmoid(output) for output in side_outputs]
            fused_prob = torch.sigmoid(fused_output)
            
            # Apply threshold to get binary edge maps
            side_edges = [(prob > threshold).float() for prob in side_probs]
            fused_edges = (fused_prob > threshold).float()
            
            # Visualize results for each image
            for j in range(images.size(0)):
                img = images[j]
                gt = targets[j]
                
                # Plot original image, ground truth and final fusion
                fig = visualize_edges(img, gt, fused_edges[j], 
                                     title=f"HED Edge Detection - {img_names[j]}")
                plt.savefig(f"hed_result_{img_names[j]}.png")
                plt.close(fig)
                
                # Plot side outputs
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Plot side outputs
                for k, side_edge in enumerate(side_edges):
                    row, col = k // 3, k % 3
                    axes[row, col].imshow(side_edge[j].squeeze().cpu().numpy(), cmap='gray')
                    axes[row, col].set_title(f"Side Output {k+1}")
                    axes[row, col].axis('off')
                
                # Plot fused output
                axes[1, 2].imshow(fused_edges[j].squeeze().cpu().numpy(), cmap='gray')
                axes[1, 2].set_title("Fused Output")
                axes[1, 2].axis('off')
                
                plt.suptitle(f"HED Side Outputs - {img_names[j]}")
                plt.tight_layout()
                plt.savefig(f"hed_side_outputs_{img_names[j]}.png")
                plt.close(fig)
                
                print(f"Processed test image {img_names[j]}")

def task4_hed_model():
    """
    Task 4: Implement Holistically Nested Edge Detection (HED)
    """
    print("Task 4: Holistically Nested Edge Detection")
    
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
    
    # Initialize model
    model = HEDModel().to(device)
    
    # Loss function
    criterion = HEDLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    # Train the model
    train_losses, val_losses, side_losses = train_hed_model(
        model, train_loader, val_loader, criterion, optimizer, num_epochs=50
    )
    
    # Evaluate on test set
    evaluate_hed_model(model, test_loader, threshold=0.5)
    
    print("Task 4 completed.")
    print("Discussion on HED model:")
    print("1. HED leverages multi-scale feature learning with side outputs from different VGG stages")
    print("2. Deep supervision with side outputs allows each layer to learn edge features")
    print("3. The fusion layer learns to weight the contribution of each side output")
    print("4. Side outputs from earlier layers capture fine details while deeper layers capture semantics")
    print("5. HED should outperform both simple CNN and VGG16 for detecting perceptually important edges")
    print("6. Comparison to Canny:")
    print("   - HED detects semantically meaningful edges while Canny detects all intensity changes")
    print("   - HED is more robust to texture and noise due to learning from human annotations")
    print("   - HED requires no parameter tuning compared to Canny's threshold selection")

if __name__ == "__main__":
    task4_hed_model()
