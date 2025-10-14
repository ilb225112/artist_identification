# Artist Identification
A Machine Learning mini-project comparing SVM and CNN approaches for artist identification from paintings. (UE23CS352A)

## Overview
This project implements and compares multiple machine learning approaches for artist identification using the Painter by Numbers dataset. The repository contains experiments with CNNs, ResNet, EfficientNet, and SVM-based pipelines, including ensemble methods for improved performance.

## Repository Structure
```
artist_identification/
├── codes/
│   ├── CNN_Experiments.ipynb          # Baseline and variant CNN experiments
│   ├── Cnn_variations.ipynb           # Additional CNN architecture experiments
│   └── SVM.ipynb                      # Feature extraction + SVM pipelines (including ensemble/fusion)
├── models/
│   ├── final/                         # Final trained model weights and pickles
│   ├── Temp/                          # Intermediate checkpoints
│   └── json_metrics_for_diff_epochs_of_cnnexp/  # Training logs and metrics
└── ml-assignment (6).ipynb            # Consolidated assignment notebook/report
```

## Dataset
This project uses the **Painter by Numbers** dataset.

- **Kaggle Dataset**: Add as input in your Kaggle notebook
- **Local Setup**: Download and place in your working directory

## Prerequisites

### Kaggle Environment (Recommended)
- GPU accelerator enabled
- Painter by Numbers dataset added as input
- Pre-installed libraries: PyTorch, torchvision, scikit-learn, numpy, pandas, matplotlib, tqdm

### Local Environment
```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib tqdm jupyter
```

## Quick Start (Kaggle)

1. **Create a new Kaggle notebook** with GPU enabled
2. **Add the Painter by Numbers dataset** as input
3. **Upload this repository's files** maintaining the directory structure
4. **Set the dataset path** in the notebook:
   ```python
   DATASET_DIR = "/kaggle/input/painter-by-numbers"
   ```
5. **Run the desired notebook**:
   - CNN experiments: `codes/CNN_Experiments.ipynb` or `codes/Cnn_variations.ipynb`
   - SVM experiments: `codes/SVM.ipynb`

## Using Pre-trained Models

Skip training and use the provided weights directly.

### Available Models
All trained models are available in:
- **Repository**: `models/final/`
- **Google Drive Mirror**: [Download Here](https://drive.google.com/drive/folders/1iDISei7-KV1EJ3UViHUGNcR1vGFDHfHl?usp=sharing)

### Model Files
- `ResNet18-Trained-Improved.pt` - ResNet18 model
- `EfficientNet-B0-Trained.pt` - EfficientNet-B0 model
- `ensemble_model.pt` - Ensemble model
- `resnet_fused_svm_model.pkl` - ResNet + SVM fusion
- `color_svm_model.pkl` - Color-based SVM model

### Loading Models

**PyTorch Models:**
```python
import torch

# Define the architecture (must match training architecture)
model = ...  

# Load weights
state = torch.load("models/final/ResNet18-Trained-Improved.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

**Scikit-learn Models:**
```python
import joblib

svm_model = joblib.load("models/final/resnet_fused_svm_model.pkl")
```

## Running Experiments

### CNN Training and Evaluation
1. Open `codes/CNN_Experiments.ipynb` or `codes/Cnn_variations.ipynb`
2. Verify the dataset path points to the Kaggle input
3. Run all cells to train/evaluate or load pre-trained weights from `models/final/`

### SVM Pipelines
1. Open `codes/SVM.ipynb`
2. Confirm feature extraction and model load paths are correct and relative
3. Run all cells to reproduce SVM baselines and ensembles

## Tips
- Always use relative paths when referencing files
- If adding Drive models as Kaggle dataset input, reference via Kaggle input mount
- Ensure architectures in code match the weight files you load
- For evaluation-only runs, disable training cells or set flags provided in the notebooks

## Troubleshooting
- **File not found**: Verify the relative path and directory structure
- **CUDA/GPU issues**: Ensure your Kaggle notebook uses GPU accelerator
- **Version mismatches**: Load models with the same library versions as used in training

## Course Information
**Course Code**: UE23CS352A  
**Project Type**: Machine Learning Mini-Project  
**Topic**: Artist Identification from Paintings

## License
This project is part of an academic assignment.
