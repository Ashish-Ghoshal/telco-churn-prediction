import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif # Import for feature selection
import joblib
from imblearn.over_sampling import SMOTE # Import SMOTE for handling class imbalance
from collections import Counter # To check class distribution

def load_data(data_file_path):
    """
    Loads the dataset from the specified CSV file path.
    Handles FileNotFoundError and other potential loading errors.

    Args:
        data_file_path (str): The full path to the CSV dataset file.

    Returns:
        pandas.DataFrame: The loaded DataFrame if successful, None otherwise.
    """
    try:
        df = pd.read_csv(data_file_path)
        print(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: '{os.path.basename(data_file_path)}' not found at {data_file_path}.")
        print("Please ensure you have uploaded the dataset to the specified directory.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

def perform_eda(df, results_dir):
    """
    Performs Exploratory Data Analysis (EDA) on the DataFrame.
    Includes basic info, descriptive statistics, missing values,
    target distribution, and feature distributions (numerical and categorical).
    It also performs initial data cleaning such as handling 'TotalCharges' spaces and mapping 'Churn'.
    Saves plots and descriptive statistics to the results directory.

    Args:
        df (pandas.DataFrame): The DataFrame to perform EDA on.
        results_dir (str): Directory to save EDA plots and summaries.

    Returns:
        pandas.DataFrame: The DataFrame after initial cleaning and EDA, ready for further preprocessing.
    """
    print("\n--- Performing Exploratory Data Analysis (EDA) ---")

    # 1. Basic Data Information
    print("\n--- Dataset Info ---")
    df.info()

    print("\n--- Descriptive Statistics (Numerical Features) ---")
    print(df.describe())
    # Save descriptive statistics to a JSON file
    descriptive_stats_path = os.path.join(results_dir, 'descriptive_statistics.json')
    df.describe().to_json(descriptive_stats_path, indent=4)
    print(f"Descriptive statistics saved to {descriptive_stats_path} ✅")

    # 2. Handle 'TotalCharges' and Missing Values (Important for consistency with old models)
    # Replace empty strings in 'TotalCharges' with NaN, then fill NaNs with 0
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    # *** IMPORTANT CHANGE: Fill NaN with 0 here, to be consistent with original notebook's likely behavior ***
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    print(f"\nHandled 'TotalCharges' spaces and filled missing values with 0. New shape: {df.shape}")
    print("\n--- Missing Values After Initial Cleaning ---")
    print(df.isnull().sum()) # Should show 0 NaNs after this step for TotalCharges

    # 3. Map 'Churn' target variable to numerical (Yes: 1, No: 0)
    if 'Churn' in df.columns and df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        print("Mapped 'Churn' column to numerical (1 for Yes, 0 for No).")


    # 4. Target Variable Distribution Plot
    print("\n--- Target Variable Distribution ---")
    print(df['Churn'].value_counts())
    print(df['Churn'].value_counts(normalize=True))

    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='viridis')
    plt.title('Distribution of Customer Churn (Target Variable)')
    plt.xlabel('Churn (0: No, 1: Yes)')
    plt.ylabel('Count')
    churn_dist_plot_path = os.path.join(results_dir, 'churn_distribution.png')
    plt.savefig(churn_dist_plot_path)
    print(f"Churn distribution plot saved to {churn_dist_plot_path} ✅")
    plt.close()

    # 5. Numerical Feature Histograms
    # 'SeniorCitizen' is int64 but represents a binary categorical (0/1). It was sometimes treated as numerical for scaling.
    numerical_features_for_eda = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Ensure SeniorCitizen is not in this list if it's explicitly treated as categorical later.
    # For histograms here, it's fine.
    plt.figure(figsize=(15, 5))
    df[numerical_features_for_eda].hist(bins=20, figsize=(15, 5), layout=(1, 3))
    plt.suptitle('Histograms of Numerical Features', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    num_hist_plot_path = os.path.join(results_dir, 'numerical_feature_histograms.png')
    plt.savefig(num_hist_plot_path)
    print(f"Numerical feature histograms saved to {num_hist_plot_path} ✅")
    plt.close()

    # 6. Categorical Feature Distributions
    # Exclude 'customerID' and 'Churn' from these plots if they are handled separately or mapped
    categorical_features_for_eda_plots = [col for col in df.columns if df[col].dtype == 'object' and col not in ['customerID', 'Churn']]
    # Include 'SeniorCitizen' for plotting as categorical if it's 0/1 (binary int)
    if 'SeniorCitizen' in df.columns and df['SeniorCitizen'].nunique() <= 2 and 'SeniorCitizen' not in categorical_features_for_eda_plots:
        categorical_features_for_eda_plots.append('SeniorCitizen')

    print(f"\nCategorical features for distribution plots: {categorical_features_for_eda_plots}")

    n_cols = 3
    n_rows = (len(categorical_features_for_eda_plots) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    axes = axes.flatten()
    for i, col in enumerate(categorical_features_for_eda_plots):
        if i < len(axes):
            sns.countplot(x=col, data=df, ax=axes[i], palette='pastel')
            axes[i].set_title(f'Distribution of {col}')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylabel('Count')
            axes[i].set_xlabel('')
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    cat_dist_plot_path = os.path.join(results_dir, 'categorical_feature_distributions.png')
    plt.savefig(cat_dist_plot_path)
    print(f"Categorical feature distributions saved to {cat_dist_plot_path} ✅")
    plt.close()

    # 7. Correlation Matrix
    # Ensure all numerical columns are considered, including Churn if it's numerical now
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix of Numerical Features and Churn')
    corr_matrix_plot_path = os.path.join(results_dir, 'correlation_matrix.png')
    plt.savefig(corr_matrix_plot_path)
    print(f"Correlation matrix plot saved to {corr_matrix_plot_path} ✅")
    plt.close()

    print("\nEDA complete. All EDA plots and descriptive statistics have been saved to your results directory.")
    return df # Return df after EDA and initial cleaning for further preprocessing


def preprocess_data(df, models_dir):
    """
    Performs full data preprocessing, including feature scaling, one-hot encoding,
    feature selection (SelectKBest), and handling class imbalance (SMOTE).
    Saves the complete preprocessing pipeline.

    Args:
        df (pandas.DataFrame): The DataFrame that has undergone initial cleaning (no NaNs, Churn mapped).
        models_dir (str): Directory to save the preprocessor pipeline.

    Returns:
        tuple: (X_train_resampled, X_test_processed, y_train_resampled, y_test, full_preprocessor_pipeline)
               The processed training and testing features, resampled training target, testing target,
               and the fitted full preprocessing pipeline.
    """
    print("\n--- Starting Data Preprocessing and Feature Engineering (Full Pipeline) ---")

    # Drop 'customerID' as it's an identifier and not a predictive feature
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("Dropped 'customerID' column.")

    # Separate features (X) and target variable (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Define numerical and categorical features for the ColumnTransformer
    # Consistent with original notebook, 'SeniorCitizen' is often included in numerical for scaling
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    # All other 'object' type columns are categorical
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numerical Features for transformation: {numerical_features}")
    print(f"Categorical Features for transformation: {categorical_features}")

    # Create the ColumnTransformer
    feature_transformer = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Split the data into training and testing sets *before* fitting the full preprocessor
    # This ensures SMOTE only sees training data and SelectKBest is properly fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Initial X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Initial X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


    # Create a pipeline that includes the ColumnTransformer and SelectKBest
    # This ensures feature selection is part of the saved preprocessor
    k_features = int(X_train.shape[1] * 0.8) # Select top 80% features
    # Apply initial transformation to get feature names for SelectKBest properly
    # This intermediate step is needed to get the correct number of features for SelectKBest
    X_train_transformed_initial = feature_transformer.fit_transform(X_train)
    
    # After initial transform, calculate k_features based on the transformed dimension
    k_features_transformed = int(X_train_transformed_initial.shape[1] * 0.8)

    # Redefine the full preprocessor pipeline to incorporate SelectKBest
    # Need to re-initialize feature_transformer if it was fitted already
    feature_transformer_for_pipeline = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    full_preprocessor_pipeline = Pipeline(steps=[
        ('features', feature_transformer_for_pipeline),
        ('selector', SelectKBest(f_classif, k=k_features_transformed))
    ])

    # Fit the full preprocessing pipeline on the training data
    # and transform both training and testing data
    X_train_processed = full_preprocessor_pipeline.fit_transform(X_train, y_train)
    X_test_processed = full_preprocessor_pipeline.transform(X_test)

    # Get feature names after all transformations for potential SHAP use
    # This requires a slightly more involved process to get names from the pipeline
    # The 'features' step is the ColumnTransformer, 'selector' is SelectKBest
    transformed_feature_names_all = full_preprocessor_pipeline.named_steps['features'].get_feature_names_out()
    selected_indices = full_preprocessor_pipeline.named_steps['selector'].get_support(indices=True)
    selected_feature_names = [transformed_feature_names_all[i] for i in selected_indices]

    print(f"Original number of features after initial transform: {X_train_transformed_initial.shape[1]}")
    print(f"Selected number of features after SelectKBest: {X_train_processed.shape[1]}")
    print(f"Selected features (first 5): {selected_feature_names[:5]}...")

    # Convert to DataFrame to apply SMOTE correctly (SMOTE works better with pandas DataFrames)
    X_train_processed_df = pd.DataFrame(X_train_processed, columns=selected_feature_names, index=X_train.index)
    X_test_processed_df = pd.DataFrame(X_test_processed, columns=selected_feature_names, index=X_test.index)


    # Handle Class Imbalance with SMOTE on the training data only
    print("\n--- Handling Class Imbalance with SMOTE ---")
    print("Original training set class distribution:", Counter(y_train))

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed_df, y_train)

    print("Resampled training set class distribution:", Counter(y_train_resampled))
    print(f"X_train_resampled shape: {X_train_resampled.shape}")
    print(f"y_train_resampled shape: {y_train_resampled.shape}")
    print("\nClass imbalance handled using SMOTE. Training data is now balanced. ✅")


    # Save the complete preprocessing pipeline (ColumnTransformer + SelectKBest)
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    joblib.dump(full_preprocessor_pipeline, preprocessor_path)
    print(f"Full preprocessing pipeline saved to {preprocessor_path} ✅")

    return X_train_resampled, X_test_processed_df, y_train_resampled, y_test, full_preprocessor_pipeline

if __name__ == '__main__':
    # This block is for testing purposes if you run data_preprocessing.py directly
    base_project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    data_raw_dir = os.path.join(base_project_path, 'data', 'raw')
    models_dir = os.path.join(base_project_path, 'models')
    results_dir = os.path.join(base_project_path, 'results')

    os.makedirs(data_raw_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Placeholder for actual data file
    data_file = os.path.join(data_raw_dir, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    df_raw = load_data(data_file)
    if df_raw is not None:
        df_eda_cleaned = perform_eda(df_raw.copy(), results_dir)
        if df_eda_cleaned is not None:
            X_train, X_test, y_train, y_test, preprocessor_pipeline = \
                preprocess_data(df_eda_cleaned.copy(), models_dir)
            print("\nFull preprocessing successful for standalone test.")
            print(f"Final X_train shape: {X_train.shape}")
            print(f"Final X_test shape: {X_test.shape}")
            print(f"Final y_train distribution: {Counter(y_train)}")

