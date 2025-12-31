import pandas as pd
from pathlib import Path
from anomalib.deploy import TorchInferencer
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
def evaluate_model(model_path, test_data_path, save_dir="evaluation_results"):
    """Evaluate a single model and generate metrics"""
    
    # Load model
    inferencer = TorchInferencer(path=model_path, device="cpu")
    
    # Collect predictions
    results = []
    test_images = list(Path(test_data_path).rglob("*.jpg")) + list(Path(test_data_path).rglob("*.png"))
    
    for img_path in test_images:
        # Get ground truth from folder structure
        gt_label = 1 if "defect" in str(img_path) else 0
        
        # Predict
        predictions = inferencer.predict(image=str(img_path))
        pred_label = predictions.pred_label
        pred_score = predictions.pred_score.item()
        
        results.append({
            "image": img_path.name,
            "ground_truth": gt_label,
            "prediction": pred_label,
            "score": pred_score,
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate metrics
    y_true = df["ground_truth"]
    y_pred = df["prediction"]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Good", "Defect"]))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f"{save_dir}/confusion_matrix.png")
    plt.close()
    
    # Save results
    df.to_csv(f"{save_dir}/predictions.csv", index=False)
    
    return df
def compare_models(old_model_path, new_model_path, test_data_path):
    """Compare performance of old vs new model"""
    
    print("=" * 80)
    print("Evaluating OLD Model")
    print("=" * 80)
    old_results = evaluate_model(old_model_path, test_data_path, "evaluation_results/old_model")
    
    print("\n" + "=" * 80)
    print("Evaluating NEW Model")
    print("=" * 80)
    new_results = evaluate_model(new_model_path, test_data_path, "evaluation_results/new_model")
    
    # Compare metrics
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    old_acc = (old_results["ground_truth"] == old_results["prediction"]).mean()
    new_acc = (new_results["ground_truth"] == new_results["prediction"]).mean()
    
    print(f"Old Model Accuracy: {old_acc:.3f}")
    print(f"New Model Accuracy: {new_acc:.3f}")
    print(f"Improvement: {(new_acc - old_acc) * 100:.2f}%")
if __name__ == "__main__":
    compare_models(
        old_model_path="results/Patchcore/socket_pins/latest/weights/torch/model.pt",
        new_model_path="results/Patchcore_improved/socket_pins/latest/weights/torch/model.pt",
        test_data_path="datasets/socket_pins/test",
    )