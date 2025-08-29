def task1_canny_edge_detection():
    """
    Task 1: Implement Canny edge detection and compare with ground truth
    """
    print("Task 1: Canny Edge Detection")
    
    # Set up transforms for loading images
    transform = transforms.Compose([
        transforms.Resize((321, 481)),  # Common size for BSDS500
        transforms.ToTensor(),
    ])
    
    # Create test dataset
    test_dataset = BSDS500Dataset(TEST_IMG_DIR, TEST_GT_DIR, transform=transform)
    
    # Test different sigma values for Gaussian blur
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    
    # Process a few sample images
    num_samples = 3
    sample_indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    for idx in sample_indices:
        image, gt_edges, img_name = test_dataset[idx]
        
        # Convert PyTorch tensor to numpy array for OpenCV processing
        img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        
        # Convert to grayscale for edge detection
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Create figure for all sigma values
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot original image
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Plot ground truth
        gt_np = gt_edges.squeeze().numpy()
        axes[0, 1].imshow(gt_np, cmap='gray')
        axes[0, 1].set_title("Ground Truth Edges")
        axes[0, 1].axis('off')
        
        # Plot Canny edges for different sigma values
        ax_idx = 2
        for i, sigma in enumerate(sigma_values):
            if i < 4:  # First 4 results go in the first row
                row, col = 0, ax_idx
            else:  # Last result goes in the second row
                row, col = 1, 0
                
            # Apply Gaussian blur with current sigma
            blurred = cv2.GaussianBlur(gray_img, (0, 0), sigma)
            
            # Apply Canny edge detection
            # We'll use automatic threshold calculation
            median_value = np.median(blurred)
            lower_threshold = int(max(0, (1.0 - 0.33) * median_value))
            upper_threshold = int(min(255, (1.0 + 0.33) * median_value))
            
            canny_edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
            
            # Plot the result
            axes[row, col].imshow(canny_edges, cmap='gray')
            axes[row, col].set_title(f"Canny (σ={sigma})")
            axes[row, col].axis('off')
            
            ax_idx = (ax_idx + 1) % 3
        
        # Add F1 score plot
        axes[1, 1].axis('off')
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Canny Edge Detection Results - Image {img_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"task1_canny_results_{img_name}.png")
        plt.show()
        
        print(f"Processed image {img_name}")
    
    print("Task 1 completed. Discussion points:")
    print("1. Canny's performance varies with sigma (blur amount)")
    print("2. Weaknesses of Canny edge detection include:")
    print("   - It detects all edges, not just perceptually important ones")
    print("   - Parameter tuning is required for each image")
    print("   - It lacks semantic understanding of the image content")
    print("   - Results depend heavily on image contrast and noise")

if __name__ == "__main__":
    task1_canny_edge_detection()
