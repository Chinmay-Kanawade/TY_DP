# TY_DP

# AutoML System with XAI (Explainable AI) Integration

This AutoML (Automated Machine Learning) system is designed for Supervised Learning tasks with a focus on providing transparency and interpretability through eXplainable AI (XAI). The system encompasses various stages of the machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), feature selection, model selection, hyperparameter optimization using Bayesian Optimization, and integrated XAI using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).

## Features
- **Data Preprocessing**: Includes removing irrelevant columns, handling duplicate values, constant values, string-numerical conversions, and missing values imputation.
- **Feature Selection**: Utilizes statistical methods like ANOVA F-value for classification and regression tasks to select the most relevant features.
- **Model Selection**: Offers a choice of decision tree-based models such as Decision Trees, Random Forest, and Gradient Boosting for both classification and regression tasks.
- **Hyperparameter Optimization**: Employs Bayesian Optimization techniques to fine-tune model hyperparameters for better performance.
- **Explainable AI (XAI)**: Provides model interpretability through SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) techniques.

## Usage
1. **Data Preprocessing**: The provided dataset undergoes preprocessing to handle irrelevant columns, duplicate values, constant values, string-numerical conversions, and missing values.
2. **Feature Selection**: Features are selected based on their relevance using statistical methods.
3. **Model Selection**: Suitable models are selected based on the type of problem (classification or regression).
4. **Hyperparameter Optimization**: Model hyperparameters are fine-tuned using Bayesian Optimization techniques.
5. **Model Evaluation**: The selected model is evaluated using various metrics such as accuracy, F1-score, precision, recall for classification tasks, and R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE) for regression tasks.
6. **Explainable AI (XAI)**: SHAP and LIME explainers are used to provide local interpretability for model predictions.

## Code Overview
The provided Python code implements the functionalities described above. It includes modules for data preprocessing, feature selection, model selection, hyperparameter optimization, and model evaluation. Additionally, it integrates SHAP and LIME explainers for model interpretability.

### Libraries Used
- NumPy
- Pandas
- Scikit-learn
- SciPy
- BayesOpt
- Matplotlib
- Seaborn
- SHAP
- LIME

### Code Structure
- **Importing Libraries**: Imports necessary libraries for data manipulation, preprocessing, modeling, and explainability.
- **Preprocessing Database**: Defines functions for data preprocessing tasks such as removing irrelevant columns, handling missing values, and encoding categorical variables.
- **Feature Selection**: Implements feature selection using statistical methods like ANOVA F-value.
- **Model Selection**: Defines functions for algorithm selection based on the type of problem (classification or regression).
- **Optimizer Functions**: Implements Bayesian Optimization for hyperparameter tuning of models.
- **Model Optimization and Training**: Integrates hyperparameter optimization and model training.
- **Model Evaluation**: Evaluates model performance using relevant metrics.
- **SHAP and LIME Explainers**: Provides local interpretability using SHAP and LIME explainers.
