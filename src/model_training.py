import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

def train_models(X_train, y_train, models_dir, results_dir):
    """
    Trains and tunes multiple classification models (Logistic Regression, SVM,
    Random Forest, XGBoost) using RandomizedSearchCV.
    Saves the best-performing models.

    Args:
        X_train (array-like): Processed training features.
        y_train (array-like): Training target variable.
        models_dir (str): Directory to save the trained models.
        results_dir (str): Directory to save model training summaries (optional).

    Returns:
        dict: A dictionary containing the best trained models.
    """
    print("\n--- Starting Model Training and Hyperparameter Tuning ---")

    models = {
        'LogisticRegression': LogisticRegression(random_state=42, solver='liblinear'),
        'SVC': SVC(random_state=42, probability=True), # probability=True is needed for ROC_AUC
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'XGBClassifier': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    }

    # Define hyperparameter distributions for RandomizedSearchCV
    param_distributions = {
        'LogisticRegression': {
            'C': uniform(loc=0, scale=4),
            'penalty': ['l1', 'l2']
        },
        'SVC': {
            'C': uniform(loc=0.1, scale=10),
            'kernel': ['linear', 'rbf']
        },
        'RandomForestClassifier': {
            'n_estimators': randint(50, 200),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 10),
            'min_samples_leaf': randint(1, 5)
        },
        'XGBClassifier': {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.3),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4)
        }
    }

    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions[name],
            n_iter=10, # Number of parameter settings that are sampled
            cv=3,       # 3-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1,  # Use all available cores
            scoring='recall' # Prioritize recall as churn prediction often focuses on identifying as many churners as possible
        )
        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        trained_models[name] = best_model

        model_path = os.path.join(models_dir, f'{name.lower()}_model.joblib')
        joblib.dump(best_model, model_path)
        print(f"Best {name} model saved to {model_path} ✅")
        print(f"Best parameters for {name}: {search.best_params_}")
        print(f"Best cross-validation score for {name}: {search.best_score_:.4f}")

    print("\nAll models trained and saved. ✅")
    return trained_models

def load_trained_models(models_dir):
    """
    Loads all trained models from the models directory.

    Args:
        models_dir (str): Directory where models are saved.

    Returns:
        dict: A dictionary containing the loaded models, or None if no models are found.
    """
    loaded_models = {}
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]

    if not model_files:
        print("No trained models found in the models directory.")
        return None

    print("\n--- Loading Trained Models ---")
    for model_file in model_files:
        try:
            name = model_file.replace('_model.joblib', '').replace('logisticregression', 'LogisticRegression').replace('svc', 'SVC').replace('randomforestclassifier', 'RandomForestClassifier').replace('xgbclassifier', 'XGBClassifier')
            model_path = os.path.join(models_dir, model_file)
            loaded_models[name] = joblib.load(model_path)
            print(f"Loaded {name} from {model_path}")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
            continue
    return loaded_models

if __name__ == '__main__':
    # This block is for testing purposes if you run model_training.py directly
    # In the full project, main.py will orchestrate this.
    base_project_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    models_dir = os.path.join(base_project_path, 'models')
    results_dir = os.path.join(base_project_path, 'results') # Not directly used here, but good practice

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Create dummy data for testing
    from sklearn.datasets import make_classification
    X_dummy, y_dummy = make_classification(n_samples=100, n_features=20, n_informative=10, n_redundant=5, random_state=42)
    # Ensure y_dummy is suitable for stratified split if used in actual preprocessing
    from sklearn.model_selection import train_test_split
    X_train_dummy, _, y_train_dummy, _ = train_test_split(X_dummy, y_dummy, test_size=0.2, random_state=42, stratify=y_dummy)


    # Check if models exist and ask for retraining
    existing_models = load_trained_models(models_dir)
    if existing_models:
        retrain_input = input("Trained models found. Do you want to retrain them? (yes/no): ").lower()
        if retrain_input == 'yes':
            print("Retraining models...")
            trained_models_result = train_models(X_train_dummy, y_train_dummy, models_dir, results_dir)
        else:
            print("Using existing models.")
            trained_models_result = existing_models
    else:
        print("No trained models found. Training new models...")
        trained_models_result = train_models(X_train_dummy, y_train_dummy, models_dir, results_dir)

    if trained_models_result:
        print("\nModel training successful for standalone test.")
        for name, model in trained_models_result.items():
            print(f"Model {name} trained: {model}")

