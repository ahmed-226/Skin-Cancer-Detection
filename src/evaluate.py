import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import cv2
import os
import joblib

def plot_confusion_matrix(y_true, y_pred, title, ax):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Non-Cancer', 'Cancer'], 
                yticklabels=['Non-Cancer', 'Cancer'], ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

def plot_roc_curve(y_true, y_prob, title, ax):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')

def plot_precision_recall_curve(y_true, y_prob, title, ax):
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    ax.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, test_files, feature_names=None):
    """Evaluate model performance with metrics and visualizations."""
    y_val_pred = model.predict(X_val)
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    print("Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['Non-Cancer', 'Cancer']))
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.3f}")
    print(f"Validation ROC-AUC: {auc(roc_curve(y_val, y_val_prob)[0], roc_curve(y_val, y_val_prob)[1]):.3f}")
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-Cancer', 'Cancer']))
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.3f}")
    print(f"Test ROC-AUC: {auc(roc_curve(y_test, y_test_prob)[0], roc_curve(y_test, y_test_prob)[1]):.3f}")

    print("\nPerforming 5-fold Cross-Validation on Training Data...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_confusion_matrix(y_val, y_val_pred, "Validation Confusion Matrix", axes[0, 0])
    plot_confusion_matrix(y_test, y_test_pred, "Test Confusion Matrix", axes[0, 1])
    plot_roc_curve(y_val, y_val_prob, "Validation ROC Curve", axes[0, 2])
    plot_roc_curve(y_test, y_test_prob, "Test ROC Curve", axes[1, 2])
    plot_precision_recall_curve(y_val, y_val_prob, "Validation Precision-Recall Curve", axes[1, 0])
    plot_precision_recall_curve(y_test, y_test_prob, "Test Precision-Recall Curve", axes[1, 1])
    plt.tight_layout()
    plt.show()

    # Make feature importance visualization more dynamic
    # Check if the model itself has feature_importances_ attribute (like Random Forest)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        if feature_names is not None:
            top_features = [feature_names[i] for i in indices]
        else:
            top_features = [f"Feature {i}" for i in indices]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=top_features)
        plt.title(f"Top 10 Feature Importances ({type(model).__name__})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    # Check for ensemble models with estimators
    elif hasattr(model, 'estimators_'):
        # For VotingClassifier, get the second estimator (usually RandomForest in your case)
        if len(model.estimators_) > 1:
            try:
                # Try to get the second estimator (index 1)
                estimator = model.estimators_[1]
                if hasattr(estimator, 'feature_importances_'):
                    importances = estimator.feature_importances_
                    indices = np.argsort(importances)[::-1][:10]
                    if feature_names is not None:
                        top_features = [feature_names[i] for i in indices]
                    else:
                        top_features = [f"Feature {i}" for i in indices]
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=importances[indices], y=top_features)
                    plt.title(f"Top 10 Feature Importances ({type(estimator).__name__})")
                    plt.xlabel("Importance")
                    plt.tight_layout()
                    plt.show()
            except (IndexError, AttributeError):
                print("Could not extract feature importances from ensemble model")

    misclassified_idx = np.where(y_test != y_test_pred)[0]
    print(f"\nNumber of misclassified test samples: {len(misclassified_idx)}")
    if len(misclassified_idx) > 0:
        print("Sample misclassified test images (indices):", misclassified_idx[:5])
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(misclassified_idx[:3]):
            img = cv2.imread(test_files[idx])
            if img is not None:
                plt.subplot(1, 3, i+1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(f"True: {['Non-Cancer', 'Cancer'][y_test[idx]]}\nPred: {['Non-Cancer', 'Cancer'][y_test_pred[idx]]}")
                plt.axis('off')
        plt.tight_layout()
        plt.show()


def evaluate_all_models(X_train, y_train, X_val, y_val, X_test, y_test, test_files, feature_names=None, output_dir="output"):

    print("\n" + "="*50)
    print("EVALUATING ALL TRAINED MODELS")
    print("="*50)
    
    # Dictionary to store model accuracy results for later comparison
    model_accuracies = {}
    
    # Look for model files in the output directory
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"No model files found in {output_dir}")
        return
    
    # Evaluate each model
    for model_file in model_files:
        model_path = os.path.join(output_dir, model_file)
        model_name = model_file.replace('_model.pkl', '').title()
        
        print(f"\n{'-'*50}")
        print(f"Evaluating {model_name} Model")
        print(f"{'-'*50}")
        
        try:
            # Load the model
            model = joblib.load(model_path)
            
            # Evaluate the model
            evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, test_files, feature_names)
            
            # Store accuracy for comparison
            y_test_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_test_pred)
            model_accuracies[model_name] = acc
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
    
    # Compare models
    if model_accuracies:
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        # Sort models by accuracy
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        for i, (model_name, acc) in enumerate(sorted_models):
            print(f"{i+1}. {model_name}: Test Accuracy = {acc:.4f}")
        
        best_model_name = sorted_models[0][0]
        print(f"\nBest performing model: {best_model_name}")
        
        # Visualize comparison
        plt.figure(figsize=(10, 6))
        model_names = [model[0] for model in sorted_models]
        accuracies = [model[1] for model in sorted_models]
        sns.barplot(x=accuracies, y=model_names)
        plt.title("Model Comparison by Test Accuracy")
        plt.xlabel("Accuracy")
        plt.xlim(min(accuracies)-0.05, 1.0)
        plt.tight_layout()
        plt.show()
