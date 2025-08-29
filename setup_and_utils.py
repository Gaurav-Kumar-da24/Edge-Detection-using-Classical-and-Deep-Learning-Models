import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
from scipy.io import loadmat
from PIL import Image
import glob
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Path settings - Update these to match your Kaggle dataset directory structure
DATA_ROOT = './BSDS500'
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train/images')
TRAIN_GT_DIR = os.path.join(DATA_ROOT, 'train/groundTruth')
VAL_IMG_DIR = os.path.join(DATA_ROOT, 'val/images')
VAL_GT_DIR = os.path.join(DATA_ROOT, 'val/groundTruth')
TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test/images')
TEST_GT_DIR = os.path.join(DATA_ROOT, 'test/groundTruth')

# Create custom dataset for BSDS500
class BSDS500Dataset(Dataset):
    def __init__(self, img_dir, gt_dir, transform=None):
        """
        Args:
            img_dir (string): Directory with all the images
            gt_dir (string): Directory with ground truth .mat files
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform
        
        # Get all image paths
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Extract image filename without extension
        img_name = os.path.basename(img_path).split('.')[0]
        
        # Load ground truth edges from .mat file
        gt_path = os.path.join(self.gt_dir, img_name + '.mat')
        gt_data = loadmat(gt_path)
        
        # BSDS500 contains multiple human annotations for each image
        # We'll use the first one for simplicity
        # In a more sophisticated approach, we could use all annotations
        gt_boundaries = gt_data['groundTruth'][0, 0]['Boundaries'][0, 0]
        
        # Convert to tensor
        gt_boundaries = torch.from_numpy(gt_boundaries.astype(np.float32))
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        # Ensure ground truth has same size as image
        gt_boundaries = gt_boundaries.unsqueeze(0)  # Add channel dimension
        
        return image, gt_boundaries, img_name

# Function to visualize results
def visualize_edges(image, gt_edges, predicted_edges, title="Edge Detection Comparison"):
    """
    Visualize original image, ground truth edges, and predicted edges side by side
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    if isinstance(image, torch.Tensor):
        # Convert tensor to numpy and denormalize if needed
        if image.shape[0] == 3:  # If image is in CHW format
            img_np = image.permute(1, 2, 0).cpu().numpy()
            # Denormalize if image was normalized during transform
            img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + 
                            np.array([0.485, 0.456, 0.406]), 0, 1)
        else:
            img_np = image.cpu().numpy()
    else:
        img_np = image
        
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Plot ground truth edges
    if isinstance(gt_edges, torch.Tensor):
        gt_np = gt_edges.squeeze().cpu().numpy()
    else:
        gt_np = gt_edges
    axes[1].imshow(gt_np, cmap='gray')
    axes[1].set_title("Ground Truth Edges")
    axes[1].axis('off')
    
    # Plot predicted edges
    if isinstance(predicted_edges, torch.Tensor):
        pred_np = predicted_edges.squeeze().cpu().numpy()
    else:
        pred_np = predicted_edges
    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title("Predicted Edges")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig

# Class balanced cross entropy loss as described in HED paper
class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BalancedCrossEntropyLoss, self).__init__()
        
    def forward(self, prediction, target):
        """
        Args:
            prediction: Tensor of shape [batch_size, 1, height, width]
            target: Tensor of shape [batch_size, 1, height, width]
        Returns:
            Balanced cross entropy loss
        """
        # Flatten prediction and target
        prediction = prediction.view(-1)
        target = target.view(-1)
        
        # Calculate positive and negative weights
        beta = torch.sum(target) / target.numel()  # Positive ratio
        
        # Avoid division by zero
        beta = torch.clamp(beta, min=0.01, max=0.99)
        
        # Positive weight (1-beta) and negative weight (beta)
        weights = torch.ones_like(target)
        weights[target > 0.5] = 1 - beta
        weights[target <= 0.5] = beta
        
        # Binary cross entropy loss
        loss = nn.functional.binary_cross_entropy_with_logits(
            prediction, target, weight=weights, reduction='mean'
        )
        
        return loss
