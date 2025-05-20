# Skin Cancer Detection

A machine learning pipeline for classifying skin lesions as cancerous or non-cancerous using dermoscopic images.

## Overview
This project implements a skin cancer detection system using traditional machine learning techniques. It includes:
- **Preprocessing**: Removes hair and glare from images.
- **Feature Extraction**: Extracts 273 features (color, texture, shape).
- **Modeling**: Trains an ensemble of SVM and Random Forest models.
- **Evaluation**: Provides metrics like accuracy, ROC-AUC, and visualizations.

The project uses the [Skin Cancer: Malignant vs. Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) dataset. Results are detailed in `docs/paper.pdf`.



## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmed-226/Skin-Cancer-Detection.git
   cd Skin-Cancer-Detection

2. Install dependencies:
    ```bash
   pip install -r requirements.txt

2. Use:
    ```bash
   jupyter notebook notebooks/main.ipynb
   ```
   or 
   ```bash
   python src/main.py
   ```