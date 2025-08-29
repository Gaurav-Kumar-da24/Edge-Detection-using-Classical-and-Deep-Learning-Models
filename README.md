# Edge Detection using Classical and Deep Learning Models 
### EE5178: Modern Computer Vision 

This project explores classical and deep learning–based methods for **edge detection** using the **BSDS500 dataset**. It implements:

- **Task 1**: Canny Edge Detection  
- **Task 2**: Simple 3-layer CNN  
- **Task 3**: VGG16-based Edge Detection (Transpose Convolution & Bilinear Upsampling)  
- **Task 4**: Holistically Nested Edge Detection (HED)

---

## 📦 Setup

### Requirements
- Python 3.6+
- PyTorch ≥ 1.7  
- Torchvision  
- OpenCV  
- NumPy  
- SciPy  
- Matplotlib  
- tqdm  
- scikit-learn  

Install dependencies:
```bash
pip install torch torchvision opencv-python scipy matplotlib numpy tqdm scikit-learn
```

### Dataset
This project uses the **Berkeley Segmentation Data Set and Benchmarks 500 (BSDS500)**.  
Download from [Berkeley Website](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html) or Kaggle.

Expected directory structure:
```
BSDS500/
├── train/
│   ├── images/
│   └── groundTruth/
├── val/
│   ├── images/
│   └── groundTruth/
└── test/
    ├── images/
    └── groundTruth/
```

---

## 📂 Code Organization
- **`setup_and_utils.py`** – Dataset class, visualization, loss functions, utilities  
- **`task1_canny_edge.py`** – Implements Canny edge detection with varying Gaussian blur (σ)  
- **`task2_simple_cnn.py`** – Defines and trains a simple 3-layer CNN for edge detection  
- **`task3_vgg16_model.py`** – Implements VGG16-based models (Transpose Convolution & Bilinear Upsampling)  
- **`task4_hed_model.py`** – Holistically Nested Edge Detection (HED) model  
- **`main.py`** – Entry script to run tasks  

---

## ▶️ Running the Code

Run **all tasks**:
```bash
python main.py --data_root ./BSDS500
```

Run a **specific task**:
```bash
# Example: Run Task 1 (Canny)
python main.py --task 1 --data_root ./BSDS500
```

---

## 📊 Task Descriptions

### **Task 1: Canny Edge Detection**
- Applies Gaussian blur with different σ values.  
- Compares results with ground-truth edges.  
- Highlights strengths/limitations of classical edge detection.

### **Task 2: Simple CNN**
- 3-layer CNN:  
  - Conv1: 3→8  
  - Conv2: 8→16  
  - Conv3: 16→1  
- Trained for 100 epochs.  
- Uses **Class-Balanced Binary Cross Entropy Loss** (per HED paper).  
- Produces binary edge maps after thresholding.

### **Task 3: VGG16 Model**
- Pretrained VGG16 as backbone (features only).  
- Two decoder variants:  
  1. **Transpose Convolution Decoder** (learnable upsampling)  
  2. **Bilinear Upsampling Decoder** (non-learnable)  
- Compared on loss curves and qualitative results.

### **Task 4: Holistically Nested Edge Detection (HED)**
- Implements side outputs from different VGG stages.  
- Uses bilinear upsampling + fusion layer.  
- Loss: multi-scale class-balanced cross entropy.  
- Produces fused edge predictions aligned with human-annotated ground truth.

---

## 📈 Results & Evaluation
The code generates:
- Side-by-side plots: **Original | Ground Truth | Prediction**  
- Training/validation **loss curves**  
- **Comparison plots**:  
  - Canny vs. CNN vs. VGG16 vs. HED  
- HED **side output visualizations** and **fusion weights**

---

## 📚 References
1. **Holistically-Nested Edge Detection** – Saining Xie, Zhuowen Tu  
2. **BSDS500 Dataset** – Berkeley Computer Vision Group  
