import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression # Import necessary for isinstance checks
from sklearn.ensemble import RandomForestClassifier # Import necessary for isinstance checks
from xgboost import XGBClassifier # Import necessary for isinstance checks
from sklearn.svm import SVC # Import necessary for isinstance checks
from sklearn.pipeline import Pipeline # Import Pipeline to correctly access steps
from sklearn.compose import ColumnTransformer # Import ColumnTransformer

# Set matplotlib backend to 'Agg' to prevent GUI issues when running non-interactively
# This must be done BEFORE importing pyplot from matplotlib if it hasn't been imported elsewhere
# If other files also import matplotlib, ensure this is set early in your main execution flow.
plt.switch_backend('Agg')

# Suppress all warnings
warnings.filterwarnings('ignore')

def make_predictions_and_interpret(model, preprocessor_pipeline, X_test, y_test, results_dir):
    """
    Makes predictions using the given model and generates SHAP explanations for interpretability.
    Saves SHAP summary and dependence plots.

    Args:
        model (object): The trained machine learning model.
        preprocessor_pipeline (object): The fitted Pipeline object (containing ColumnTransformer and SelectKBest).
        X_test (pandas.DataFrame): Original (unprocessed) testing features, needed for feature names.
        y_test (array-like): Testing target variable.
        results_dir (str): Directory to save SHAP plots.
    """
    print("\n--- Starting Prediction and Model Interpretation (SHAP) ---")

    # Transform X_test using the full preprocessor pipeline before prediction
    # This X_test_transformed will be a NumPy array with the correct number of features
    X_test_transformed_numpy = preprocessor_pipeline.transform(X_test)

    # Make predictions
    y_pred = model.predict(X_test_transformed_numpy)
    print(f"Predictions made using the best model: {model.__class__.__name__}")

    # SHAP for Model Interpretability
    print("\nGenerating SHAP plots for model interpretability...")

    # Reconstruct feature names after preprocessing pipeline
    column_transformer = preprocessor_pipeline.named_steps['features']

    # Get feature names from ColumnTransformer's transformers, handling scikit-learn version differences
    try:
        numerical_feature_names_from_col_trans = column_transformer.named_transformers_['num'].get_feature_names_out(column_transformer.transformers_[0][2])
    except AttributeError:
        numerical_feature_names_from_col_trans = column_transformer.transformers_[0][2]

    try:
        categorical_feature_names_from_col_trans = column_transformer.named_transformers_['cat'].get_feature_names_out(column_transformer.transformers_[1][2])
    except AttributeError:
        categorical_feature_names_from_col_trans = column_transformer.named_transformers_['cat'].get_feature_names(column_transformer.transformers_[1][2])

    # Combine numerical and categorical feature names after initial transformation
    all_transformed_feature_names = list(numerical_feature_names_from_col_trans) + list(categorical_feature_names_from_col_trans)

    # Apply feature selection to get the final list of feature names that correspond to X_test_transformed_numpy
    selector = preprocessor_pipeline.named_steps['selector']
    selected_feature_indices = selector.get_support(indices=True)
    final_feature_names = [all_transformed_feature_names[i] for i in selected_feature_indices]

    # Create a DataFrame from the transformed NumPy array with correct feature names
    # This ensures SHAP plotting functions receive named columns for easier mapping.
    X_test_transformed_df_for_shap = pd.DataFrame(X_test_transformed_numpy, columns=final_feature_names, index=X_test.index)


    # Initialize SHAP explainer based on model type
    if isinstance(model, (RandomForestClassifier, XGBClassifier)):
        explainer = shap.TreeExplainer(model)
        # For tree models, shap_values can be directly computed
        shap_values = explainer.shap_values(X_test_transformed_df_for_shap) # Pass DataFrame here
    elif isinstance(model, (LogisticRegression, SVC)):
        # KernelExplainer needs a background dataset (sample from training data, or here, test data for simplicity)
        # Pass DataFrame to KernelExplainer for background data consistency with main data
        background_data_sample_df = shap.utils.sample(X_test_transformed_df_for_shap, min(100, X_test_transformed_df_for_shap.shape[0]), random_state=42)
        
        explainer = shap.KernelExplainer(model.predict_proba, background_data_sample_df)
        # Pass DataFrame for SHAP value computation
        shap_values = explainer.shap_values(X_test_transformed_df_for_shap)
        print(f"  Note: Using KernelExplainer for {model.__class__.__name__}. This may take some time.")
    else:
        print(f"SHAP explainer for {model.__class__.__name__} is not explicitly supported or optimized. Skipping SHAP plots.")
        return

    # For classification, shap_values typically returns a list of arrays (one for each class)
    # For binary classification, we usually look at shap_values[1] for the positive class (churn=1)
    if isinstance(shap_values, list):
        shap_values_for_positive_class = shap_values[1]
    else:
        shap_values_for_positive_class = shap_values # For models that return single array

    # 1. SHAP Summary Plot (Global Feature Importance)
    plt.figure(figsize=(10, 8))
    # Pass the DataFrame with correct column names for shap.summary_plot
    shap.summary_plot(shap_values_for_positive_class, X_test_transformed_df_for_shap, show=False)
    shap_summary_path = os.path.join(results_dir, f'shap_summary_plot_{model.__class__.__name__.lower()}.png')
    plt.savefig(shap_summary_path, bbox_inches='tight')
    print(f"SHAP summary plot saved to {shap_summary_path} ✅")
    plt.close()

    # 2. SHAP Dependence Plots (for top features)
    # Get top 5 features by mean absolute SHAP value
    feature_importance_dict = {
        final_feature_names[i]: np.mean(np.abs(shap_values_for_positive_class[:, i]))
        for i in range(shap_values_for_positive_class.shape[1])
    }
    sorted_features = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)
    top_n_features = [feat[0] for feat in sorted_features[:5]]

    print(f"\nGenerating SHAP dependence plots for top 5 features: {', '.join(top_n_features)}...")
    for feature_name_to_plot in top_n_features:
        try:
            plt.figure(figsize=(8, 6))
            # IMPORTANT FIX: Using feature name (string) as first arg, with DataFrame X
            shap.dependence_plot(
                feature_name_to_plot, # <--- Reverted this argument to string name, consistent with Colab
                shap_values_for_positive_class,
                X_test_transformed_df_for_shap, # Keep the full DataFrame
                interaction_index=None,
                show=False
            )
            # Replace special characters in feature name for filename
            clean_feature_name = feature_name_to_plot.replace(" ", "_").replace("__", "_").replace("[", "").replace("]", "").lower()
            shap_dependence_path = os.path.join(results_dir, f'shap_dependence_plot_{clean_feature_name}_{model.__class__.__name__.lower()}.png')
            plt.savefig(shap_dependence_path, bbox_inches='tight')
            print(f"SHAP dependence plot for '{feature_name_to_plot}' saved to {shap_dependence_path} ✅")
            plt.close()
        except Exception as e:
            print(f"Could not generate SHAP dependence plot for {feature_name_to_plot}: {e}")


    print("\n--- Prediction and Model Interpretation Complete ---")

if __name__ == '__main__':
    # This block is for testing purposes if you run predict.py directly
    base_project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    models_dir = os.path.join(base_project_path, 'models')
    results_dir = os.path.join(base_project_path, 'results')

    os.makedirs(results_dir, exist_ok=True)

    # Dummy data and preprocessor for testing
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_selection import SelectKBest, f_classif
    from imblearn.over_sampling import SMOTE # For dummy data simulation

    # Create dummy raw data that resembles the actual dataset structure
    n_samples = 200
    n_original_features = 19 # Corresponds to columns before OHE/SelectKBest
    X_dummy_raw_np, y_dummy = make_classification(n_samples=n_samples, n_features=n_original_features-4, n_informative=8, n_redundant=2, random_state=42)
    
    # Create a DataFrame for X_dummy_raw to simulate original data with categorical and numerical features
    dummy_columns = [f'num_feat_{i}' for i in range(n_original_features-4)] # numerical features
    X_dummy_raw = pd.DataFrame(X_dummy_raw_np, columns=dummy_columns)
    
    # Add dummy categorical features
    X_dummy_raw['cat_feat_A'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    X_dummy_raw['cat_feat_B'] = np.random.choice(['X', 'Y'], size=n_samples)
    X_dummy_raw['cat_feat_C'] = np.random.choice(['P', 'Q', 'R'], size=n_samples)
    X_dummy_raw['SeniorCitizen'] = np.random.choice([0, 1], size=n_samples) # Simulating SeniorCitizen as numerical for ColumnTransformer

    # Simulate preprocessing pipeline:
    numerical_features_dummy = [col for col in X_dummy_raw.columns if X_dummy_raw[col].dtype != 'object'] # All non-object dtypes
    categorical_features_dummy = [col for col in X_dummy_raw.columns if X_dummy_raw[col].dtype == 'object'] # All object dtypes

    # ColumnTransformer
    feature_transformer_dummy = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_dummy),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_dummy)
        ],
        remainder='passthrough'
    )

    # Initial transform to get feature count for SelectKBest calculation
    X_temp_transformed_initial = feature_transformer_dummy.fit_transform(X_dummy_raw)
    k_features_transformed_dummy = int(X_temp_transformed_initial.shape[1] * 0.8) # Select top 80%

    # Full Pipeline including ColumnTransformer and SelectKBest
    preprocessor_pipeline_dummy = Pipeline(steps=[
        ('features', feature_transformer_dummy),
        ('selector', SelectKBest(f_classif, k=k_features_transformed_dummy))
    ])

    X_train_dummy, X_test_dummy, y_train_dummy, y_test_dummy = train_test_split(
        X_dummy_raw, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy
    )

    # Fit the preprocessor pipeline on training data and transform both train/test
    X_train_processed_dummy = preprocessor_pipeline_dummy.fit_transform(X_train_dummy, y_train_dummy)
    X_test_processed_dummy = preprocessor_pipeline_dummy.transform(X_test_dummy)

    # Simulate SMOTE on training data
    smote_dummy = SMOTE(random_state=42)
    X_train_resampled_dummy, y_train_resampled_dummy = smote_dummy.fit_resample(X_train_processed_dummy, y_train_dummy)

    # "Train" a dummy model (e.g., Logistic Regression) on resampled data
    dummy_model = LogisticRegression(random_state=42, solver='liblinear').fit(X_train_resampled_dummy, y_train_resampled_dummy)

    print("Running prediction and interpretation with dummy data and model...")
    # Pass original X_test_dummy (DataFrame) and the full preprocessor_pipeline_dummy
    make_predictions_and_interpret(dummy_model, preprocessor_pipeline_dummy, X_test_dummy, y_test_dummy, results_dir)
    print("\nPrediction and interpretation successful for standalone test.")
