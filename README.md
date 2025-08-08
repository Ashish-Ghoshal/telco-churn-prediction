# Telco Customer Churn Prediction: A Binary Classification Project 📉

## Table of Contents

1.  [Overview](#overview)
2.  [Key Features](#key-features)
3.  [Technologies and Libraries Used](#technologies-and-libraries-used)
4.  [Project Structure](#project-structure)
5.  [Setup and Execution](#setup-and-execution)
    * [1. Clone the Repository](#1-clone-the-repository)
    * [2. Download the Dataset](#2-download-the-dataset)
    * [3. Create and Activate Conda Environment](#3-create-and-activate-conda-environment)
    * [4. Install Dependencies](#4-install-dependencies)
    * [5. Run the Project](#5-run-the-project)
6.  [How to Use and Interpret Results](#how-to-use-and-interpret-results)
    * [Interpreting Results](#interpreting-results)
7.  [Future Enhancements](#future-enhancements)
8.  [Contributing](#contributing)
9.  [License](#license) 

## Overview

# 

This project focuses on building a robust machine learning solution to **predict customer churn** within a telecommunications company. Customer churn, the phenomenon where customers stop using a company's services, is a significant challenge for businesses, directly impacting revenue and growth. By leveraging historical customer data, this project develops and evaluates various classification models to identify customers who are highly likely to churn. The insights gained from these predictions enable the company to implement **proactive, targeted retention strategies**, thereby minimizing customer attrition and enhancing long-term profitability. This repository showcases a complete, end-to-end machine learning pipeline, encompassing detailed data preprocessing, comprehensive exploratory data analysis (EDA), rigorous model training with hyperparameter tuning, in-depth performance evaluation, and crucial model interpretability using SHAP.

## Key Features

# 

This project provides a comprehensive set of functionalities designed for effective churn prediction:

*   **Data Preprocessing**: A meticulously crafted pipeline handles common data issues, including the identification and treatment of missing values, appropriate conversion of data types (e.g., converting 'TotalCharges' from object to numeric), robust scaling of numerical features using `StandardScaler`, and efficient encoding of categorical variables via `OneHotEncoder`. This ensures the data is in an optimal format for model consumption.
    
*   **Exploratory Data Analysis (EDA)**: Extensive visual and statistical analyses are performed to gain deep insights into the dataset. This includes visualizing data distributions, uncovering correlations between various customer attributes, and explicitly examining the relationship between different service types, contract durations, and the likelihood of churn.
    
*   **Multiple Classification Models**: To ensure a robust and comparative analysis, the project implements and rigorously compares the performance of several widely-used machine learning classification models. These include **Logistic Regression**, **Support Vector Machines (SVM)**, **Random Forest**, and **XGBoost classifiers**, providing a well-rounded view of different algorithmic strengths.
    
*   **Hyperparameter Tuning**: Model performance is meticulously optimized through the application of **RandomizedSearchCV**. This technique efficiently explores a predefined hyperparameter space to identify the best combination of parameters for each model, maximizing their predictive accuracy and generalization capabilities.
    
*   **Comprehensive Evaluation**: The trained models undergo a thorough evaluation using a suite of standard classification metrics. This includes a **confusion matrix** (to visualize True Positives, True Negatives, False Positives, False Negatives), **accuracy**, **precision**, **recall**, **F1-score**, **ROC-AUC curve** (Receiver Operating Characteristic - Area Under Curve), and **Precision-Recall curve**. These metrics provide a holistic view of model effectiveness across different performance aspects.
    
*   **Threshold Optimization**: Beyond default settings, the project discusses and demonstrates a pragmatic approach to choosing an optimal classification threshold. This critical step allows for tailoring the model's output to specific business needs, for instance, prioritizing recall to minimize missed churners even if it means a slight increase in false positives.
    
*   **Model** Interpretability **(SHAP)**: To move beyond black-box predictions, **SHAP (SHapley Additive exPlanations)** is integrated. This powerful technique provides transparent insights into feature importance, revealing which factors are most influential in the model's overall predictions and how individual features contribute to specific churn predictions.
    
*   **Modular Codebase**: The entire codebase is designed with modularity in mind, organized into reusable Python scripts (`data_preprocessing.py`, `model_training.py`, `evaluation.py`, `predict.py`). This enhances clarity, facilitates maintenance, and promotes code reusability for future projects.
    
*   **Google Colab Workflow**: For ease of use, experimentation, and cloud-based execution, a dedicated, step-by-step Google Colab notebook is provided. This notebook mirrors the local project structure, allowing users to quickly set up, run, and explore the pipeline without complex local configurations.
    

## Technologies and Libraries Used

# 

This project leverages a standard Python data science stack for machine learning development:

*   **Python 3.9+**: The core programming language for the entire project.
    
*   **Data Manipulation**:
    
    *   `pandas`: Indispensable for high-performance data manipulation and analysis, especially with tabular data.
        
    *   `numpy`: Provides powerful numerical computing capabilities for array operations and mathematical functions.
        
*   **Machine Learning**:
    
    *   `scikit-learn`: A comprehensive library offering a wide range of machine learning algorithms, preprocessing tools, and evaluation metrics.
        
    *   `xgboost`: A highly optimized gradient boosting library known for its speed and predictive power, especially on structured data.
        
*   **Data Visualization**:
    
    *   `matplotlib`: A fundamental plotting library for creating static, interactive, and animated visualizations in Python.
        
    *   `seaborn`: Built on top of matplotlib, it provides a high-level interface for drawing attractive and informative statistical graphics.
        
*   **Model Interpretability**:
    
    *   `shap`: A cutting-edge library for explaining the output of any machine learning model, providing global and local interpretability.
        
*   **Serialization**:
    
    *   `joblib`: Efficiently saves and loads Python objects, particularly useful for persisting trained machine learning models and preprocessing pipelines.
        

## Project Structure

# 

The repository is organized into a clear and intuitive directory structure to promote maintainability and collaboration.

    .
    ├── data/
    │   └── raw/
    │       └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Raw dataset: This is where the downloaded dataset should be placed.
    ├── notebooks/
    │   └── Telco_Churn_Prediction_Colab.ipynb  # Google Colab notebook (conceptual): A high-level reference for the Colab workflow.
    ├── src/                                  # Source code directory for modular Python scripts.
    │   ├── __init__.py                       # Initializes the 'src' directory as a Python package.
    │   ├── data_preprocessing.py             # Handles data loading, cleaning, feature engineering, and train-test split.
    │   ├── model_training.py                 # Contains functions for training various ML models and hyperparameter tuning.
    │   ├── evaluation.py                     # Responsible for model evaluation, generating metrics, and creating plots.
    │   └── predict.py                        # Manages making predictions and generating SHAP interpretability insights.
    ├── models/                               # Directory to save trained machine learning models (e.g., .joblib files).
    ├── results/                              # Directory to store evaluation reports, performance metrics, and visualization plots.
    ├── main.py                               # The primary script to orchestrate and run the entire ML pipeline from start to finish.
    ├── requirements.txt                      # Lists all necessary Python dependencies required to run the project.
    └── README.md                             # This comprehensive documentation file, providing project details and instructions.
    
    

## Setup and Execution

# 

Follow these step-by-step instructions to set up the project environment and execute the machine learning pipeline locally on your machine.

### 1\. Clone the Repository

# 

Begin by cloning this repository to your local machine using Git. Open your terminal or command prompt and run:

    git clone https://github.com/Ashish-Ghoshal/telco-churn-prediction.git
    cd telco-churn-prediction
    
    

### 2\. Download the Dataset

# 

The project relies on the Telco Customer Churn dataset. Please download it from its official source:

Telco Customer Churn Dataset on Kaggle

Once downloaded, place the WA\_Fn-UseC\_-Telco-Customer-Churn.csv file into the designated data/raw/ directory within your newly cloned repository. Ensure the file name is exact.

### 3\. Create and Activate Conda Environment

# 

It is **highly recommended** to use a dedicated Conda environment to manage project dependencies. This isolates the project's requirements, preventing conflicts with other Python projects on your system.

    conda create -n telco_churn_env python=3.9
    conda activate telco_churn_env
    
    

This command creates a new Conda environment named `telco_churn_env` with Python 3.9 and then activates it, making it your active Python environment for this project.

### 4\. Install Dependencies

# 

With your Conda environment activated, install all the necessary Python libraries listed in `requirements.txt`.

    pip install -r requirements.txt
    
    

This command will automatically download and install all required packages (pandas, numpy, scikit-learn, xgboost, shap, joblib, matplotlib, seaborn, imbalanced-learn).

### 5\. Run the Project

# 

Once all dependencies are installed, you can execute the entire machine learning pipeline by running the main script.

    python main.py
    
    

This script will sequentially perform the following actions:

*   **Load and Preprocess Data**: It will read the `WA_Fn-UseC_-Telco-Customer-Churn.csv` dataset, handle missing values, convert data types, perform feature scaling, and encode categorical variables, preparing the data for model training.
    
*   **Train and Tune Models**: It will train Logistic Regression, SVM, Random Forest, and XGBoost models, applying `RandomizedSearchCV` for effective hyperparameter tuning to optimize each model's performance.
    
*   **Evaluate Models**: All trained models will be thoroughly evaluated on unseen test data, generating a suite of performance metrics including accuracy, precision, recall, F1-score, and ROC-AUC.
    
*   **Generate SHAP Plots**: For the best-performing model, SHAP (SHapley Additive exPlanations) plots will be generated to provide critical insights into feature importance and how various factors influence churn predictions.
    
*   **Save Outputs**: The trained machine learning models (as `.joblib` files) will be saved to the `models/` directory. All evaluation results, classification reports, confusion matrices, and insightful plots (like ROC-AUC and SHAP visualizations) will be saved to the `results/` directory for easy access and review.
    

## How to Use and Interpret Results

# 

After successfully running `main.py`, the `models/` and `results/` directories will be populated with the outputs of the machine learning pipeline.

*   **`models/` directory**: This directory contains the serialized (`.joblib`) files for all trained models (e.g., `logistic_regression_model.joblib`, `random_forest_model.joblib`). These files allow you to load and reuse the trained models without retraining them. The `preprocessor.joblib` file is also saved here, crucial for preprocessing new data before making predictions with the saved models.
    
*   **`results/` directory**: This is where all the analytical outputs and visualizations are stored:
    
    *   `classification_summary.csv`: A concise CSV file providing a comparative overview of key evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC) for all trained models.
        
    *   `classification_report_[model_name].txt`: Detailed text files, one for each model, containing a comprehensive classification report that breaks down precision, recall, and F1-score per class (churn vs. no-churn), along with support.
        
    *   `confusion_matrix_[model_name].png`: Visual representations of the confusion matrix for each model, showing the counts of True Positives, True Negatives, False Positives, and False Negatives. These are vital for understanding the types of errors made by each model.
        
    *   `roc_auc_curve_all_models.png`: A single plot displaying the ROC-AUC curves for all models on the same graph, allowing for easy visual comparison of their ability to discriminate between churners and non-churners across different classification thresholds.
        
    *   `precision_recall_curve_all_models.png`: A single plot showing the Precision-Recall curves for all models, which is particularly informative for imbalanced datasets and when balancing the cost of False Positives and False Negatives is important.
        
    *   `shap_summary_plot_[model_name].png`: A global SHAP summary plot for the best-performing model. This plot illustrates the overall feature importance, showing which features have the most significant impact on the model's predictions and whether they tend to drive predictions towards or away from churn.
        
    *   `shap_dependence_plot_[feature_name]_[model_name].png`: Individual SHAP dependence plots for the top influential features of the best model. These plots reveal how the value of a single feature impacts the prediction, often highlighting non-linear relationships or interactions.
        

### Interpreting Results: Data, Model Performance, and Strategic Choices

#### Data Characteristics: Addressing Churn Imbalance

# 

The initial Exploratory Data Analysis (EDA) reveals that the dataset exhibits a **class imbalance**, with approximately **73.5% of customers being non-churners (`No`) and 26.5% being churners (`Yes`)**. While not as extreme as some real-world datasets, this imbalance is significant enough to warrant careful consideration during model training and evaluation. To mitigate this, **SMOTE (Synthetic Minority Over-sampling Technique)** is applied during preprocessing. SMOTE synthetically increases the number of instances in the minority class (churners) in the training data, ensuring models are not biased towards the majority class and can learn more effectively from churn patterns.

#### Model Performance: A Focus on Recall

# 

Upon evaluating the models, we observe the following performance metrics from `classification_summary.csv`:

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score (Churn) | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| LogisticRegression | 0.738 | 0.504 | 0.805 | 0.620 | 0.840 |
| RandomForestClassifier | 0.768 | 0.550 | 0.690 | 0.612 | 0.836 |
| randomforest | 0.771 | 0.569 | 0.575 | 0.572 | 0.821 |
| SVC | 0.749 | 0.521 | 0.687 | 0.593 | 0.799 |
| svm | 0.751 | 0.523 | 0.687 | 0.594 | 0.797 |
| XGBClassifier | 0.732 | 0.504 | 0.805 | 0.613 | 0.838 |
| xgboost | 0.777 | 0.574 | 0.631 | 0.601 | 0.834 |

In a churn prediction scenario, **minimizing False Negatives (missed churners)** is often paramount. A false negative means a customer who was going to churn was incorrectly predicted as a non-churner, resulting in a lost opportunity for intervention and retention. Conversely, a False Positive (predicting churn when a customer wouldn't have churned) might lead to unnecessary retention efforts but doesn't directly result in a lost customer. Therefore, we strategically **prioritize Recall for the 'Churn' (positive) class (class 1)**.

Both **Logistic Regression** and **XGBClassifier** models demonstrate excellent recall, successfully identifying approximately **80.5%** of actual churners in the test set. While other models might show slightly higher overall accuracy, their high recall directly aligns with the business objective of minimizing customer attrition. The choice between these top-performing models can depend on factors like interpretability and computational cost.

You'll notice a trade-off between Precision and Recall. Models with higher recall for churn might have slightly lower precision (meaning more false positives). The ultimate choice depends on the business's specific cost-benefit analysis of False Positives vs. False Negatives.

#### Optimal Threshold Value: Balancing Business Needs

# 

The default classification threshold for binary classification models is typically 0.5. However, in scenarios like churn prediction where one class (churn) is more critical, **tuning this threshold is essential**. For each model, an **optimal threshold** is calculated with the objective of achieving approximately **70% recall for the churn class**.

Here are the optimal thresholds found for each model, extracted from `optimal_thresholds_summary.json`:

*   **LogisticRegression**: `0.6062`
    
*   **RandomForestClassifier**: `0.4803`
    
*   **randomforest**: `0.3850`
    
*   **SVC**: `0.4320`
    
*   **svm**: `0.4243`
    
*   **XGBClassifier**: `0.5711`
    
*   **xgboost**: `0.4082`
    

For your chosen **Logistic Regression** model, an optimal threshold of **0.6062** was determined. This means that if the model predicts a churn probability greater than 0.6062, it classifies the customer as a churner. This tailored threshold allows the business to:

*   **Adjust Sensitivity**: Increase or decrease the model's sensitivity to churn based on the cost of false positives vs. false negatives.
    
*   **Optimize Interventions**: Ensure that retention efforts are targeted at a balance between capturing most at-risk customers (high recall) and avoiding too many unnecessary interventions (acceptable precision).
    

The ROC-AUC and Precision-Recall curves (saved as PNGs in `results/`) further illustrate these trade-offs and help visualize how models perform across different thresholds.

#### SHAP Plots: Explaining Why Customers Churn

# 

SHAP (SHapley Additive exPlanations) provides deep insights into the model's decisions. The generated SHAP summary plot for the Logistic Regression model, for instance, helps visualize global feature importance.

This plot illustrates the overall impact of each feature on the model's output (churn probability). Key observations from such a plot typically include:

*   **Feature Importance**: Features like `tenure`, `TotalCharges`, `Contract_Two year`, `InternetService_Fiber optic`, and `Contract_One year` are often among the most influential factors.
    
*   **Direction of Impact**: Red dots (higher feature values) pushed the prediction towards churn for features like `MonthlyCharges` or `InternetService_Fiber optic`, while blue dots (lower feature values) pushed predictions away from churn for features like `tenure` or `Contract_Two year`. This indicates that shorter tenure, higher monthly charges, and certain internet service/contract types contribute significantly to increased churn risk.
    

While the individual dependence plots are not yet successfully generated, the summary plot already offers crucial insights into what factors the model considers most important for churn.

## Future Enhancements

# 

To make this project even more robust and valuable in a real-world context, consider:

*   **Advanced Feature Engineering**: Explore creating interaction terms (e.g., `MonthlyCharges / tenure`) and more sophisticated categorical encodings.
    
*   **Hyperparameter Optimization**: Implement **Bayesian Optimization** (e.g., `Optuna`) for more efficient tuning.
    
*   **Ensemble Methods**: Experiment with **LightGBM**, **CatBoost**, and **model stacking** for potentially higher performance.
    
*   **Deployment**: Containerize the application with **Docker** and deploy it as a **REST API** using **FastAPI** for real-time predictions.
    
*   **MLOps**: Integrate **MLflow** for experiment tracking and **DVC** for data/model versioning to manage the lifecycle effectively.
    
*   **Dashboard**: Develop an interactive dashboard (e.g., with **Streamlit**) for real-time insights and user interaction.
    

## Contributing

# 

Contributions are warmly welcome! If you have suggestions for improvements, new features, or find any bugs, please feel free to open an issue or submit a pull request.

Here's a general workflow for contributing:

1.  **Fork the repository**: Click the 'Fork' button at the top right of the GitHub page to create your copy of this repository.
    
2.  **Create a new branch**: From your local clone, create a new branch for your feature or bug fix:
    
        git checkout -b feature/your-feature-name-or-bugfix-description
        
        
    
    (e.g., `git checkout -b feature/add-lightgbm-model` or `git checkout -b bugfix/fix-data-loading`)
    
3.  **Make your changes**: Implement your new features or bug fixes. Ensure your code adheres to the existing coding style and includes appropriate comments and docstrings.
    
4.  **Test** your **changes**: Thoroughly test your modifications to ensure they work as expected and don't introduce new issues.
    
5.  **Commit your changes**: Commit your changes with a clear and concise commit message:
    
        git commit -m 'Add new feature: [Brief description]'
        
        
    
    (e.g., `git commit -m 'Feat: Implement LightGBM model training'`)
    
6.  **Push to the branch**: Push your local branch to your forked repository on GitHub:
    
        git push origin feature/your-feature-name-or-bugfix-description
        
        
    
7.  **Open a pull request**: Go to your forked repository on GitHub, and you will see an option to open a Pull Request to the original repository. Provide a detailed description of your changes and why they are valuable.
    

## License

# 

This project is licensed under the MIT License - see the `LICENSE` file in the repository for full details. This open-source license allows you to freely use, modify, and distribute the code, provided you include the original license.