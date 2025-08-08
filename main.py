import os
import sys
import joblib
import pandas as pd # Needed for X_test_original_for_shap slicing
from sklearn.model_selection import train_test_split # Used here for consistent data splitting
from collections import Counter # For checking class distribution

# Add src directory to Python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import functions from custom modules
from data_preprocessing import load_data, perform_eda, preprocess_data
from model_training import train_models, load_trained_models
from evaluation import evaluate_models
from predict import make_predictions_and_interpret

def main():
    """
    Main function to orchestrate the Telco Customer Churn Prediction pipeline.
    This script will:
    1. Set up project directories.
    2. Load and perform EDA on the dataset, including initial data cleaning.
    3. Based on user input, either load existing models and the full preprocessing pipeline,
       or train new models and fit a new preprocessing pipeline.
    4. Evaluate the trained models.
    5. Make predictions and generate SHAP interpretability plots for the best model.
    """
    print("--- Starting Telco Customer Churn Prediction Pipeline ---")

    # Define base project path and create necessary directories
    base_project_path = os.getcwd()
    data_raw_dir = os.path.join(base_project_path, 'data', 'raw')
    models_dir = os.path.join(base_project_path, 'models')
    results_dir = os.path.join(base_project_path, 'results')

    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print(f"Project base path: {base_project_path}")
    print(f"Data directory: {data_raw_dir}")
    print(f"Models directory: {models_dir}")
    print(f"Results directory: {results_dir}")

    # --- Step 1: Load Data ---
    data_file_path = os.path.join(data_raw_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_raw = load_data(data_file_path)

    if df_raw is None:
        print("Exiting pipeline due to data loading error.")
        return

    # --- Step 2: Perform EDA (and initial cleaning of TotalCharges and Churn mapping) ---
    # `perform_eda` now ensures `TotalCharges` NaNs are filled with 0 and `Churn` is mapped,
    # and returns this cleaned DataFrame.
    df_eda_cleaned = perform_eda(df_raw.copy(), results_dir) # Pass a copy for EDA visualizations

    if df_eda_cleaned is None:
        print("Exiting pipeline as EDA and initial cleaning failed.")
        return

    # Define original X and y from the *cleaned* DataFrame for consistent splitting
    original_X_for_split = df_eda_cleaned.drop('Churn', axis=1)
    original_y_for_split = df_eda_cleaned['Churn']

    # Perform train-test split once on the cleaned original data.
    # X_train_original_df and X_test_original_for_shap are DataFrames
    # This is crucial for consistent data input to the preprocessor.
    X_train_original_df, X_test_original_for_shap_df, y_train, y_test = \
        train_test_split(original_X_for_split, original_y_for_split, test_size=0.2, random_state=42, stratify=original_y_for_split)

    # Initialize variables
    X_train_processed = None
    X_test_processed = None
    preprocessor_pipeline = None # This will hold the full pipeline (ColumnTransformer + SelectKBest)
    trained_models = None

    # --- Step 3 & 4: Model Training / Loading and Corresponding Preprocessing ---
    existing_models_found = load_trained_models(models_dir) # Check if models exist

    retrain_choice = None
    if existing_models_found:
        while True:
            retrain_input = input("Trained models found. Do you want to retrain them? (yes/no): ").lower().strip()
            if retrain_input in ['yes', 'no']:
                retrain_choice = retrain_input
                break
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
    else:
        print("No trained models found. Will proceed with training new models.")
        retrain_choice = 'yes' # Force retraining if no models exist

    if retrain_choice == 'yes':
        print("\nProceeding with training new models and fitting a new full preprocessing pipeline...")
        # preprocess_data will now return X_train_resampled, X_test_processed, y_train_resampled, y_test, and the fitted pipeline
        X_train_processed, X_test_processed, y_train_resampled, y_test_unchanged, preprocessor_pipeline = \
            preprocess_data(df_eda_cleaned.copy(), models_dir) # Pass original cleaned df for preprocessing
        
        # Update y_train to be the resampled one
        y_train = y_train_resampled
        # y_test should remain unchanged
        # y_test = y_test_unchanged # No need to reassign, it's already from the initial split

        trained_models = train_models(X_train_processed, y_train, models_dir, results_dir)
    else: # retrain_choice == 'no'
        print("\nLoading existing models and the full preprocessing pipeline...")
        preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        if os.path.exists(preprocessor_path):
            preprocessor_pipeline = joblib.load(preprocessor_path)
            print(f"Full preprocessing pipeline loaded from {preprocessor_path} ✅")
        else:
            print("Error: Saved preprocessor pipeline (preprocessor.joblib) not found alongside existing models.")
            print("Please ensure 'preprocessor.joblib' exists in the 'models/' directory or choose to retrain.")
            return

        trained_models = existing_models_found

        # Transform the *original* X_train and X_test using the LOADED preprocessor_pipeline
        # X_train_original_df needs to be transformed to match the feature space of loaded models.
        # Note: SMOTE is not part of the `preprocessor_pipeline` but is applied after its output in new training.
        # If loading, the models are trained on SMOTE'd data, so we don't re-SMOTE here.
        X_train_processed = preprocessor_pipeline.transform(X_train_original_df)
        X_test_processed = preprocessor_pipeline.transform(X_test_original_for_shap_df)
        
        # If loading models, the y_train is the original y_train from the split, not resampled
        # The models themselves are already trained on SMOTEd data implicitly if they were trained with SMOTE.
        # This means y_train remains original_y_for_split from the initial split
        # We need to ensure that the y_train passed to evaluation is the original one,
        # and if the loaded model implies SMOTE, it means it handles class imbalance internally.

    if trained_models is None or not trained_models:
        print("Exiting pipeline as no models are available for evaluation or prediction.")
        return

    # --- Step 5: Model Evaluation ---
    # Pass y_train from the *initial* split for evaluation, as X_train_processed is for model training.
    # Evaluation metrics should reflect performance on the *un-resampled* test set.
    evaluation_summary_df = evaluate_models(trained_models, X_test_processed, y_test, results_dir)
    print("\n--- Evaluation Summary ---")
    print(evaluation_summary_df)

    # --- Step 6: Prediction and Interpretation (using the best model) ---
    if not evaluation_summary_df.empty:
        if 'Recall (Churn)' in evaluation_summary_df.columns:
            best_model_name = evaluation_summary_df.loc[evaluation_summary_df['Recall (Churn)'].idxmax()]['Model']
            best_model = trained_models[best_model_name]
            print(f"\nSelected '{best_model_name}' as the best model for interpretation (based on Recall).")
        else:
            print("Warning: 'Recall (Churn)' column not found. Selecting best model by ROC-AUC if available.")
            if 'ROC-AUC' in evaluation_summary_df.columns:
                best_model_name = evaluation_summary_df.loc[evaluation_summary_df['ROC-AUC'].idxmax()]['Model']
                best_model = trained_models[best_model_name]
                print(f"\nSelected '{best_model_name}' as the best model for interpretation (based on ROC-AUC).")
            else:
                print("No suitable metric found for best model selection. Falling back to XGBClassifier if available.")
                if 'XGBClassifier' in trained_models:
                    best_model_name = 'XGBClassifier'
                    best_model = trained_models['XGBClassifier']
                    print(f"Falling back to '{best_model_name}' for interpretation.")
                else:
                    print("No suitable model found for interpretation. Exiting.")
                    return
    else:
        print("Could not determine best model for interpretation as evaluation summary is empty. Exiting.")
        return

    # Pass the original X_test DataFrame portion for SHAP feature naming,
    # and the full preprocessor_pipeline for internal transformation.
    make_predictions_and_interpret(best_model, preprocessor_pipeline, X_test_original_for_shap_df, y_test, results_dir)

    print("\n--- Telco Customer Churn Prediction Pipeline Complete! ---")

if __name__ == '__main__':
    main()

