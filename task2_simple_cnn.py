class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Simple 3-layer CNN as specified in the assignment
        # Conv1: Input channels=3 (RGB), Output channels=8, kernel_size=3, padding=1
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        # Conv2: Input channels=8, Output channels=16, kernel_size=3, padding=1
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Conv3: Input channels=16, Output channels=1 (edge map), kernel_size=3, padding=1
        self.conv3 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        # No activation after last layer - we'll apply sigmoid when needed
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)  # No activation here - will be applied in the loss function
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_name):
    """
    Train the model and validate
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
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
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update statistics
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), f"{model_name}.pth")
    
    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{model_name}_loss.png")
    plt.show()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, threshold=0.5, model_name="Model"):
    """
    Evaluate the model on test data and visualize results
    """
    model.eval()
    
    # Process a few test samples
    with torch.no_grad():
        for i, (images, targets, img_names) in enumerate(test_loader):
            if i >= 5:  # Limit to 5 test images
                break
                
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            
            # Apply threshold to get binary edge map
            pred_edges = (probs > threshold).float()
            
            # Visualize results
            for j in range(images.size(0)):
                img = images[j]
                gt = targets[j]
                pred = pred_edges[j]
                
                fig = visualize_edges(img, gt, pred, 
                                      title=f"{model_name} Edge Detection - {img_names[j]}")
                plt.savefig(f"{model_name}_result_{img_names[j]}.png")
                plt.close(fig)
                
                print(f"Processed test image {img_names[j]}")

def task2_simple_cnn():
    """
    Task 2: Train a simple 3-layer CNN for edge detection
    """
    print("Task 2: Simple CNN Model")
    
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
    model = SimpleCNN().to(device)
    
    # Loss function - using the class balanced cross entropy loss
    criterion = BalancedCrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=100, model_name="simple_cnn"
    )
    
    # Evaluate on test set
    evaluate_model(model, test_loader, threshold=0.5, model_name="SimpleCNN")
    
    print("Task 2 completed.")
    print("Discussion on loss function:")
    print("1. We used class-balanced binary cross entropy loss which addresses the imbalance")
    print("   between edge and non-edge pixels (typically ~90% of pixels are non-edges).")
    print("2. This loss function assigns higher weights to the minority class (edge pixels)")
    print("   and lower weights to the majority class (non-edge pixels).")
    print("3. No activation function was used in the output layer during training since")
    print("   binary_cross_entropy_with_logits is more numerically stable.")
    print("4. For inference, we apply sigmoid to get probabilities and then threshold.")

if __name__ == "__main__":
    task2_simple_cnn()
