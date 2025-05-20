import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocessing import process_image
from src.extract_features import extract_all_features

def extract_and_save_features(train_files, test_files, train_labels, test_labels, output_path='output/features.npz'):
    """Process all images, extract features, and save to NPZ file."""
    print(f"Processing {len(train_files)} training images and {len(test_files)} test images...")
    train_features = []
    test_features = []
    
    for i, img_path in enumerate(train_files):
        if i % 10 == 0:
            print(f"Processing training image {i+1}/{len(train_files)}")
        processed_img = process_image(img_path)
        if processed_img:
            features = extract_all_features(processed_img['glare_removed'])
            train_features.append(features)
    
    for i, img_path in enumerate(test_files):
        if i % 10 == 0:
            print(f"Processing test image {i+1}/{len(test_files)}")
        processed_img = process_image(img_path)
        if processed_img:
            features = extract_all_features(processed_img['glare_removed'])
            test_features.append(features)
    
    train_features = np.array(train_features)
    test_features = np.array(test_features)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        train_features=train_features,
        test_features=test_features,
        train_labels=np.array(train_labels[:len(train_features)]),
        test_labels=np.array(test_labels[:len(test_features)])
    )
    
    print(f"Features extracted and saved to {output_path}")
    print(f"Training features shape: {train_features.shape}")
    print(f"Testing features shape: {test_features.shape}")
    
    return train_features, test_features

def train_models(X_train, y_train):
    """Train SVM, Random Forest, and ensemble models with grid search."""
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    svm = SVC(probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=3, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_
    print("Best SVM params:", svm_grid.best_params_)
    
    rf = RandomForestClassifier()
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_
    print("Best Random Forest params:", rf_grid.best_params_)
    
    ensemble = VotingClassifier(estimators=[
        ('svm', best_svm),
        ('rf', best_rf)
    ], voting='soft')
    ensemble.fit(X_train, y_train)
    
    os.makedirs("output", exist_ok=True)
    joblib.dump(best_svm, "output/svm_model.pkl")
    joblib.dump(best_rf, "output/rf_model.pkl")
    joblib.dump(ensemble, "output/ensemble_model.pkl")
    
    return best_svm, best_rf, ensemble