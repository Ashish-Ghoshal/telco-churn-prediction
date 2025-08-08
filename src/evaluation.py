import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import json # For saving optimal thresholds

def evaluate_models(models, X_test, y_test, results_dir):
    """
    Evaluates trained models using various classification metrics and generates plots.
    Saves classification reports, confusion matrices, and ROC-AUC/Precision-Recall curves.
    Also calculates and saves optimal thresholds for recall.

    Args:
        models (dict): A dictionary of trained model objects.
        X_test (array-like): Processed testing features.
        y_test (array-like): Testing target variable.
        results_dir (str): Directory to save evaluation reports and plots.

    Returns:
        pandas.DataFrame: A summary DataFrame of evaluation metrics for all models.
    """
    print("\n--- Starting Model Evaluation ---")

    evaluation_summary = []
    fpr_tpr_data = {} # To store FPR, TPR for ROC-AUC curve plot
    precision_recall_data = {} # To store Precision, Recall for PR curve plot
    optimal_thresholds = {}

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        # Predictions
        y_pred = model.predict(X_test)
        # Predict probabilities for ROC-AUC and Precision-Recall curves
        if hasattr(model, "predict_proba"):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"): # For SVC when probability=True is not set
            # For SVC, if probability=True is not set, decision_function values need to be scaled
            # to be interpretable as probabilities. This is a common workaround.
            # However, in model_training.py, SVC is set with probability=True.
            decision_scores = model.decision_function(X_test)
            y_pred_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
        else:
            print(f"Model {name} does not have predict_proba or decision_function. Skipping probability-based metrics.")
            y_pred_proba = None

        # 1. Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        report_path = os.path.join(results_dir, f'classification_report_{name.lower().replace(" ", "")}.txt')
        with open(report_path, 'w') as f:
            f.write(classification_report(y_test, y_pred))
        print(f"Classification report saved to {report_path} ✅")

        # Extract metrics for summary
        accuracy = report['accuracy']
        precision = report['1']['precision'] # Precision for churn (positive class)
        recall = report['1']['recall']       # Recall for churn (positive class)
        f1_score = report['1']['f1-score']   # F1-score for churn (positive class)

        roc_auc = 0
        if y_pred_proba is not None:
            # 2. ROC-AUC Curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            print(f"ROC-AUC: {roc_auc:.4f}")
            fpr_tpr_data[name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'roc_auc': roc_auc}

            # 3. Precision-Recall Curve
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_vals, precision_vals)
            print(f"Precision-Recall AUC: {pr_auc:.4f}")
            precision_recall_data[name] = {'precision': precision_vals.tolist(), 'recall': recall_vals.tolist(), 'pr_auc': pr_auc}

            # 4. Optimal Threshold Calculation (prioritizing recall)
            max_recall_threshold = 0
            best_f1_score_at_threshold = 0
            opt_thresh = 0.5 # Default threshold
            
            if len(pr_thresholds) > 0:
                # Find the threshold that maximizes recall while maintaining a reasonable precision
                # A simple approach: find threshold closest to 0.7 recall
                target_recall = 0.7
                idx = np.argmin(np.abs(recall_vals - target_recall))
                if idx < len(pr_thresholds): # Ensure index is valid
                    opt_thresh = pr_thresholds[idx]
                    print(f"Optimal threshold for {name} (prioritizing ~{target_recall*100}% recall): {opt_thresh:.4f}")
                else:
                    opt_thresh = 0.5
                    print(f"Could not find optimal threshold for {name} prioritizing {target_recall*100}% recall, using default 0.5.")
            else:
                opt_thresh = 0.5
                print(f"No valid PR thresholds for {name}, using default 0.5.")

            # FIX: Convert numpy.float values to standard Python float for JSON serialization
            optimal_thresholds[name] = float(opt_thresh)

        # 5. Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {name}')
        cm_plot_path = os.path.join(results_dir, f'confusion_matrix_{name.lower().replace(" ", "")}.png')
        plt.savefig(cm_plot_path)
        print(f"Confusion matrix plot saved for {name} ✅")
        plt.close()

        evaluation_summary.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision (Churn)': precision,
            'Recall (Churn)': recall,
            'F1-Score (Churn)': f1_score,
            'ROC-AUC': roc_auc
        })

    # Save optimal thresholds to a JSON file
    optimal_thresholds_path = os.path.join(results_dir, 'optimal_thresholds_summary.json')
    with open(optimal_thresholds_path, 'w') as f:
        json.dump(optimal_thresholds, f, indent=4)
    print(f"Optimal thresholds summary saved to {optimal_thresholds_path} ✅")


    # Combine ROC-AUC curves on a single plot
    plt.figure(figsize=(10, 8))
    for name, data in fpr_tpr_data.items():
        plt.plot(data['fpr'], data['tpr'], label=f'{name} (AUC = {data["roc_auc"]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC-AUC Curve for All Models')
    plt.legend(loc='lower right')
    roc_auc_plot_path = os.path.join(results_dir, 'roc_auc_curve_all_models.png')
    plt.savefig(roc_auc_plot_path)
    print(f"Combined ROC-AUC curve plot saved to {roc_auc_plot_path} ✅")
    plt.close()

    # Combine Precision-Recall curves on a single plot
    plt.figure(figsize=(10, 8))
    for name, data in precision_recall_data.items():
        plt.plot(data['recall'], data['precision'], label=f'{name} (AUC = {data["pr_auc"]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for All Models')
    plt.legend(loc='lower left')
    pr_curve_plot_path = os.path.join(results_dir, 'precision_recall_curve_all_models.png')
    plt.savefig(pr_curve_plot_path)
    print(f"Combined Precision-Recall curve plot saved to {pr_curve_plot_path} ✅")
    plt.close()

    evaluation_df = pd.DataFrame(evaluation_summary)
    summary_path = os.path.join(results_dir, 'classification_summary.csv')
    evaluation_df.to_csv(summary_path, index=False)
    print(f"\nEvaluation summary saved to {summary_path} ✅")
    print("\n--- Model Evaluation Complete ---")
    return evaluation_df

if __name__ == '__main__':
    # This block is for testing purposes if you run evaluation.py directly
    base_project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    models_dir = os.path.join(base_project_path, 'models')
    results_dir = os.path.join(base_project_path, 'results')

    os.makedirs(results_dir, exist_ok=True)

    # Dummy data and models for testing
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.svm import SVC

    # Create dummy data with more features to simulate your actual dataset size
    X_dummy, y_dummy = make_classification(n_samples=200, n_features=40, n_informative=20, n_redundant=10, random_state=42)
    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy)

    # Create and "train" dummy models
    dummy_log_reg = LogisticRegression(random_state=42, solver='liblinear').fit(X_train_dummy, y_train_dummy)
    dummy_rf = RandomForestClassifier(random_state=42).fit(X_train_dummy, y_train_dummy)
    dummy_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_train_dummy, y_train_dummy)
    dummy_svc = SVC(random_state=42, probability=True).fit(X_train_dummy, y_train_dummy)

    dummy_models = {
        'LogisticRegression': dummy_log_reg,
        'SVC': dummy_svc,
        'RandomForestClassifier': dummy_rf,
        'XGBClassifier': dummy_xgb
    }

    print("Running evaluation with dummy data and models...")
    evaluation_df = evaluate_models(dummy_models, X_test_dummy, y_test_dummy, results_dir)
    print("\nEvaluation successful for standalone test.")
    print(evaluation_df)
