"""
Model training script for the MCA AI mini project.

This script trains a Decision Tree Classifier using dataset.csv and prints
accuracy, classification report, feature importance, and a sample prediction.
Run:
    python model_training.py
"""

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

DATASET_PATH = "dataset.csv"
FEATURE_COLUMNS = [
    "Attendance_Percentage",
    "Internal_Assessment_Marks",
    "Assignment_Score",
    "Study_Hours_Per_Day",
    "Previous_Semester_Percentage",
]
TARGET_COLUMN = "Performance"


def load_data() -> pd.DataFrame:
    """Read the dataset from CSV."""
    return pd.read_csv(DATASET_PATH)


def train_decision_tree(df: pd.DataFrame):
    """Train and evaluate the Decision Tree Classifier."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=3,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, X_test, y_test, predictions, accuracy


def main():
    """Execute model training and print evaluation details."""
    df = load_data()
    model, X_test, y_test, predictions, accuracy = train_decision_tree(df)

    print("AI-Based Student Performance Prediction System")
    print("Algorithm: Decision Tree Classifier")
    print("Dataset records:", len(df))
    print(f"Prediction Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Feature Importance:")
    for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

    sample_student = pd.DataFrame(
        [[85, 42, 88, 4.5, 82]],
        columns=FEATURE_COLUMNS,
    )
    sample_prediction = model.predict(sample_student)[0]
    print("\nSample Student Input:")
    print(sample_student.to_string(index=False))
    print("Sample Prediction:", sample_prediction)


if __name__ == "__main__":
    main()
