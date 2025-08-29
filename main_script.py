import os
import torch
import argparse

# Import task functions
from task1_canny_edge import task1_canny_edge_detection
from task2_simple_cnn import task2_simple_cnn
from task3_vgg16_model import task3_vgg16_model
from task4_hed_model import task4_hed_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Edge Detection Assignment')
    parser.add_argument('--task', type=int, default=0, 
                        help='Task to run (1-4), 0 for all tasks')
    parser.add_argument('--data_root', type=str, default='./BSDS500',
                        help='Path to BSDS500 dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    args = parser.parse_args()
    
    # Update global variables
    global DATA_ROOT, TRAIN_IMG_DIR, TRAIN_GT_DIR, VAL_IMG_DIR, VAL_GT_DIR, TEST_IMG_DIR, TEST_GT_DIR
    DATA_ROOT = args.data_root
    TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'train/images')
    TRAIN_GT_DIR = os.path.join(DATA_ROOT, 'train/groundTruth')
    VAL_IMG_DIR = os.path.join(DATA_ROOT, 'val/images')
    VAL_GT_DIR = os.path.join(DATA_ROOT, 'val/groundTruth')
    TEST_IMG_DIR = os.path.join(DATA_ROOT, 'test/images')
    TEST_GT_DIR = os.path.join(DATA_ROOT, 'test/groundTruth')
    
    # Print system information
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run selected task(s)
    if args.task == 0 or args.task == 1:
        task1_canny_edge_detection()
    
    if args.task == 0 or args.task == 2:
        task2_simple_cnn()
    
    if args.task == 0 or args.task == 3:
        task3_vgg16_model()
    
    if args.task == 0 or args.task == 4:
        task4_hed_model()
    
    print("All tasks completed!")

if __name__ == "__main__":
    main()
