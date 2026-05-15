## Case Study: Predicting Employee Attrition
## BANA 620 — Predictive Analytics & Data Mining
##
## Instructions:
##   Use Claude Code to complete each TODO section.
##   Refer to the Claude Code prompts in employee_attrition.ipynb for guidance.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Data URL ──────────────────────────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
COLS_TO_DROP = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']

NUMERICAL_FEATURES = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
    'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
    'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears',
    'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
    'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

CATEGORICAL_FEATURES = [
    'BusinessTravel', 'Department', 'EducationField',
    'Gender', 'JobRole', 'MaritalStatus', 'OverTime'
]


def load_data(url: str) -> pd.DataFrame:
    """Load the HR Employee Attrition dataset from a GitHub raw URL.

    Parameters
    ----------
    url : str
        Raw GitHub URL pointing to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset with original columns intact.
    """
    # TODO: Use pd.read_csv() to load the dataset from the URL
    pass


def create_attrition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create Table 1: attrition rate (%) by Department and Job Role.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with 'Department', 'JobRole', and 'Attrition' columns.

    Returns
    -------
    pd.DataFrame
        Summary with columns: Department, Job Role, Attrition Rate (%)
        Sorted descending by Attrition Rate (%).
    """
    # TODO: groupby(['Department', 'JobRole']) on the Attrition column
    # Calculate the percentage of 'Yes' values (multiply mean by 100, round to 1 decimal)
    # Rename columns: Department, Job Role, Attrition Rate (%)
    # Sort from highest to lowest attrition rate
    pass


def clean_and_encode(df: pd.DataFrame):
    """Prepare features for modeling.

    Steps:
    1. Drop COLS_TO_DROP (non-predictive columns)
    2. Encode target: Attrition → 1 (Yes) or 0 (No)
    3. One-hot encode CATEGORICAL_FEATURES with drop_first=True

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X (feature matrix) and y (target vector).
    """
    df = df.copy()
    # TODO: Drop COLS_TO_DROP from df
    # TODO: Encode Attrition: (df['Attrition'] == 'Yes').astype(int)
    # TODO: Assign y = df['Attrition'] and X = df.drop(columns=['Attrition'])
    # TODO: One-hot encode CATEGORICAL_FEATURES using pd.get_dummies(drop_first=True)
    # TODO: return X, y
    pass


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    """Split data 80/20 and standardize numerical features.

    Parameters
    ----------
    X : pd.DataFrame
        Encoded feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, scaler
    """
    # TODO: train_test_split with test_size=0.2 and random_state=42
    # TODO: Instantiate StandardScaler
    # TODO: Identify numerical columns present in X_train
    #       (intersection of NUMERICAL_FEATURES and X_train.columns)
    # TODO: fit_transform on X_train numerical cols, transform on X_test
    # TODO: return X_train, X_test, y_train, y_test, scaler
    pass


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train Logistic Regression, Decision Tree, and Random Forest.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    dict
        {'Logistic Regression': fitted_model,
         'Decision Tree': fitted_model,
         'Random Forest': fitted_model}
    """
    # TODO: Instantiate LogisticRegression(max_iter=1000, random_state=42)
    # TODO: Instantiate DecisionTreeClassifier(random_state=42)
    # TODO: Instantiate RandomForestClassifier(n_estimators=100, random_state=42)
    # TODO: Fit all three on X_train, y_train
    # TODO: Return dictionary {name: model}
    pass


def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Evaluate all models and return Table 2.

    Parameters
    ----------
    models : dict
        {model_name: fitted_model}

    Returns
    -------
    pd.DataFrame
        Columns: Model, Training Accuracy, Testing Accuracy, F1 Score, AUROC
        All metrics rounded to 4 decimal places.
    """
    # TODO: For each model in models.items():
    #   - predict on X_train → train_pred
    #   - predict on X_test  → test_pred
    #   - predict_proba on X_test → test_proba[:, 1]
    #   - compute accuracy_score, f1_score, roc_auc_score
    # TODO: Build rows list and return pd.DataFrame
    pass


def get_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """Extract top 10 feature importances from a fitted Random Forest.

    Parameters
    ----------
    model : fitted RandomForestClassifier
    feature_names : list
        Column names from X_train (use list(X_train.columns)).

    Returns
    -------
    pd.DataFrame
        Top 10 features sorted by importance descending.
        Columns: Feature, Importance (rounded to 4 decimal places).
    """
    # TODO: Access model.feature_importances_
    # TODO: Create DataFrame with Feature and Importance columns
    # TODO: Sort descending, round to 4 decimals, return top 10
    pass
