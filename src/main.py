
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.preprocessing import process_image
from src.train_model import extract_and_save_features, train_models
from src.evaluate import evaluate_model

def load_data(data_dir="data/Skin_Data"):
    """Load image file paths and labels."""
    cancer_train_dir = os.path.join(data_dir, 'Cancer', 'Training')
    cancer_test_dir = os.path.join(data_dir, 'Cancer', 'Testing')
    non_cancer_train_dir = os.path.join(data_dir, 'Non_Cancer', 'Training')
    non_cancer_test_dir = os.path.join(data_dir, 'Non_Cancer', 'Testing')

    cancer_train_files = [os.path.join(cancer_train_dir, f) for f in os.listdir(cancer_train_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    non_cancer_train_files = [os.path.join(non_cancer_train_dir, f) for f in os.listdir(non_cancer_train_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    cancer_test_files = [os.path.join(cancer_test_dir, f) for f in os.listdir(cancer_test_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
    non_cancer_test_files = [os.path.join(non_cancer_test_dir, f) for f in os.listdir(non_cancer_test_dir) if f.lower().endswith(('.jpg', '.jpeg'))]

    train_labels_cancer = [1] * len(cancer_train_files)
    train_labels_non_cancer = [0] * len(non_cancer_train_files)
    test_labels_cancer = [1] * len(cancer_test_files)
    test_labels_non_cancer = [0] * len(non_cancer_test_files)

    train_files = cancer_train_files + non_cancer_train_files
    train_labels = train_labels_cancer + train_labels_non_cancer
    test_files = cancer_test_files + non_cancer_test_files
    test_labels = test_labels_cancer + test_labels_non_cancer

    print("Number of Cancer training files:", len(cancer_train_files))
    print("Number of Non-Cancer training files:", len(non_cancer_train_files))
    print("Number of Cancer testing files:", len(cancer_test_files))
    print("Number of Non-Cancer testing files:", len(non_cancer_test_files))

    return train_files, train_labels, test_files, test_labels



def main():
    
    data_dir = "data/Skin_Data"  
    train_files, train_labels, test_files, test_labels = load_data(data_dir)

    
    features_file = "output/skin_cancer_features.npz"
    if not os.path.exists(features_file):
        train_features, test_features = extract_and_save_features(
            train_files, test_files, train_labels, test_labels, features_file
        )
    else:
        print(f"Loading features from {features_file}")
        data = np.load(features_file)
        train_features = data['train_features']
        test_features = data['test_features']
        train_labels = data['train_labels']
        test_labels = data['test_labels']

    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    
    X_train, X_val, y_train, y_val = train_test_split(
        train_features_scaled, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    
    best_svm, best_rf, ensemble = train_models(X_train, y_train)

    
    feature_names = (
        [f"Color_Hist_R_{i}" for i in range(64)] +
        [f"Color_Hist_G_{i}" for i in range(64)] +
        [f"Color_Hist_B_{i}" for i in range(64)] +
        [f"Color_Moment_{m}_{c}" for c in ['R', 'G', 'B'] for m in ['Mean', 'Variance', 'Skewness', 'Kurtosis']] +
        [f"HSV_{s}_{c}" for c in ['H', 'S', 'V'] for s in ['Mean', 'Std']] +
        [f"LAB_{s}_{c}" for c in ['L', 'A', 'B'] for s in ['Mean', 'Std']] +
        [f"Haralick_{p}_{i}" for p in ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation', 'ASM'] for i in range(1, 5)] +
        [f"Haralick_Mahotas_{i}" for i in [0, 1, 2, 3, 4, 8]] +
        ['Intensity_Mean', 'Intensity_Std', 'Intensity_Entropy', 'Intensity_Kurtosis'] +
        [f"Wavelet_{c}_{s}" for c in ['H', 'V', 'D'] for s in ['Mean', 'Std']] +
        [f"LBP_{i}" for i in range(26)] +
        [f"Hu_Moment_{i}" for i in range(7)] +
        ['Diameter', 'Circularity', 'Asymmetry', 'Border_Irregularity']
    )

    
    print("\nEvaluating Ensemble Model:")
    evaluate_model(ensemble, X_train, y_train, X_val, y_val, test_features_scaled, test_labels, test_files, feature_names)
    
    
    print("\nEvaluating All Trained Models:")
    from src.evaluate import evaluate_all_models
    evaluate_all_models(X_train, y_train, X_val, y_val, test_features_scaled, test_labels, test_files, feature_names, output_dir="output")

if __name__ == "__main__":
    main()