## Case Study: Predicting Employee Attrition — INSTRUCTOR SOLUTION
## BANA 620 — Predictive Analytics & Data Mining

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
    """Load the HR Employee Attrition dataset from a GitHub raw URL."""
    df = pd.read_csv(url)
    return df


def create_attrition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create Table 1: attrition rate (%) by Department and Job Role."""
    summary = (
        df.groupby(['Department', 'JobRole'])['Attrition']
        .apply(lambda x: (x == 'Yes').mean() * 100)
        .round(1)
        .reset_index()
    )
    summary.columns = ['Department', 'Job Role', 'Attrition Rate (%)']
    summary = summary.sort_values('Attrition Rate (%)', ascending=False).reset_index(drop=True)
    return summary


def clean_and_encode(df: pd.DataFrame):
    """Prepare features for modeling."""
    df = df.copy()
    df = df.drop(columns=COLS_TO_DROP)
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    y = df['Attrition']
    X = df.drop(columns=['Attrition'])
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.Series):
    """Split 80/20 and standardize numerical features."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    num_cols = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test, scaler


def train_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Train Logistic Regression, Decision Tree, and Random Forest."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def evaluate_models(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Evaluate all models and return Table 2."""
    rows = []
    for name, model in models.items():
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        rows.append({
            'Model': name,
            'Training Accuracy': round(accuracy_score(y_train, train_pred), 4),
            'Testing Accuracy': round(accuracy_score(y_test, test_pred), 4),
            'F1 Score': round(f1_score(y_test, test_pred), 4),
            'AUROC': round(roc_auc_score(y_test, test_proba), 4),
        })
    return pd.DataFrame(rows)


def get_feature_importances(model, feature_names: list) -> pd.DataFrame:
    """Extract top 10 feature importances from a fitted Random Forest."""
    fi = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    })
    fi = fi.sort_values('Importance', ascending=False).head(10)
    fi['Importance'] = fi['Importance'].round(4)
    fi = fi.reset_index(drop=True)
    return fi


# ── Main Execution ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Load
    df = load_data(DATA_URL)
    print(f"Dataset shape: {df.shape}")
    print(f"Attrition distribution:\n{df['Attrition'].value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")

    # Table 1
    print("\n── Table 1: Attrition Rate by Department & Job Role ──")
    table1 = create_attrition_table(df)
    print(table1.to_string(index=False))

    # Encode and split
    X, y = clean_and_encode(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # Train
    models = train_models(X_train, y_train)

    # Table 2
    print("\n── Table 2: Model Comparison ──")
    results_df = evaluate_models(models, X_train, X_test, y_train, y_test)
    print(results_df.to_string(index=False))

    # Table 3
    rf_model = models['Random Forest']
    fi_df = get_feature_importances(rf_model, list(X_train.columns))
    print("\n── Table 3: Top 10 Feature Importances ──")
    print(fi_df.to_string(index=False))

    # Feature importance bar chart
    plt.figure(figsize=(8, 5))
    sns.barplot(data=fi_df, x='Importance', y='Feature', palette='Blues_r')
    plt.title('Top 10 Features Predicting Employee Attrition')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
