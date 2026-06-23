"""
AI-Based Student Performance Prediction System Using Decision Tree Algorithm
MCA Artificial Intelligence Mini Project

Run command:
    streamlit run app.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
CLASS_ORDER = ["Excellent", "Good", "Average", "Poor"]


@st.cache_data
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load the student performance dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            "dataset.csv was not found. Keep dataset.csv in the same folder as app.py."
        )
    return pd.read_csv(path)


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Train a Decision Tree Classifier and return model assets."""
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

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, X_train, X_test, y_train, y_test, y_pred, accuracy


def performance_message(label: str) -> str:
    """Return a student-friendly interpretation for each predicted class."""
    messages = {
        "Excellent": "The student is performing at a very strong level and is likely to maintain high academic outcomes.",
        "Good": "The student is performing well, with scope for improvement through consistent practice and revision.",
        "Average": "The student may need more structured study time, better assignment quality, and stronger internal assessment preparation.",
        "Poor": "The student requires immediate academic support, attendance improvement, and a guided study plan.",
    }
    return messages.get(label, "Prediction generated successfully.")


def build_prediction_input(
    attendance: float,
    internal_marks: float,
    assignment_score: float,
    study_hours: float,
    previous_percentage: float,
) -> pd.DataFrame:
    """Create a one-row DataFrame matching the model training schema."""
    return pd.DataFrame(
        [[attendance, internal_marks, assignment_score, study_hours, previous_percentage]],
        columns=FEATURE_COLUMNS,
    )


def plot_feature_importance(model: DecisionTreeClassifier):
    """Create a bar chart of feature importance values from the trained tree."""
    importance_df = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#64748b", "#0f766e", "#2563eb", "#ca8a04", "#dc2626"]
    ax.barh(importance_df["Feature"], importance_df["Importance"], color=colors)
    ax.set_title("Feature Importance - Decision Tree Classifier", fontsize=13, weight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Input Feature")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_decision_tree(model: DecisionTreeClassifier):
    """Create a visualization of the trained Decision Tree model."""
    fig, ax = plt.subplots(figsize=(20, 10))
    tree.plot_tree(
        model,
        feature_names=FEATURE_COLUMNS,
        class_names=list(model.classes_),
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree Visualization", fontsize=16, weight="bold", pad=20)
    fig.tight_layout()
    return fig


def plot_performance_distribution(df: pd.DataFrame):
    """Create a chart showing the count of students in each performance class."""
    counts = df[TARGET_COLUMN].value_counts().reindex(CLASS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(counts.index, counts.values, color=["#15803d", "#2563eb", "#ca8a04", "#dc2626"])
    ax.set_title("Student Performance Distribution", fontsize=13, weight="bold")
    ax.set_xlabel("Performance Category")
    ax.set_ylabel("Number of Students")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred):
    """Create a confusion matrix heatmap using Matplotlib."""
    labels = CLASS_ORDER
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(6.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=13, weight="bold")
    ax.set_xlabel("Predicted Performance")
    ax.set_ylabel("Actual Performance")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, matrix[i, j], ha="center", va="center", color="#111827")

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="AI Student Performance Prediction",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .main .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        .metric-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 16px;
            background: #ffffff;
        }
        .prediction-box {
            border-radius: 8px;
            padding: 20px;
            border: 1px solid #d1d5db;
            background: #f8fafc;
        }
        .small-note {color: #475569; font-size: 0.95rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("AI-Based Student Performance Prediction System")
    st.caption("MCA Artificial Intelligence Mini Project using Decision Tree Classification")

    df = load_dataset()
    model, X_train, X_test, y_train, y_test, y_pred, accuracy = train_model(df)

    with st.sidebar:
        st.header("Prediction Panel")
        st.write("Enter student academic details to predict performance.")

        attendance = st.slider("Attendance Percentage", 0.0, 100.0, 78.0, 1.0)
        internal_marks = st.slider("Internal Assessment Marks", 0.0, 50.0, 35.0, 1.0)
        assignment_score = st.slider("Assignment Score", 0.0, 100.0, 75.0, 1.0)
        study_hours = st.slider("Study Hours Per Day", 0.0, 10.0, 3.0, 0.5)
        previous_percentage = st.slider("Previous Semester Percentage", 0.0, 100.0, 72.0, 1.0)

        predict_button = st.button("Predict Performance", type="primary", use_container_width=True)

    total_students = len(df)
    train_size = len(X_train)
    test_size = len(X_test)
    class_count = df[TARGET_COLUMN].nunique()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Prediction Accuracy", f"{accuracy * 100:.2f}%")
    metric_cols[1].metric("Dataset Records", total_students)
    metric_cols[2].metric("Training Samples", train_size)
    metric_cols[3].metric("Testing Samples", test_size)

    st.divider()

    left_col, right_col = st.columns([0.95, 1.05], gap="large")

    with left_col:
        st.subheader("Student Prediction Result")
        input_df = build_prediction_input(
            attendance,
            internal_marks,
            assignment_score,
            study_hours,
            previous_percentage,
        )

        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        probability_df = pd.DataFrame(
            {
                "Performance": model.classes_,
                "Confidence": probabilities,
            }
        ).sort_values("Confidence", ascending=False)

        if predict_button:
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2 style="margin-top:0; color:#0f172a;">Predicted Performance: {prediction}</h2>
                    <p class="small-note">{performance_message(prediction)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Adjust the input values and click Predict Performance to view the AI prediction.")

        st.write("Model confidence by class")
        st.dataframe(
            probability_df.assign(Confidence=lambda data: (data["Confidence"] * 100).round(2)),
            hide_index=True,
            use_container_width=True,
        )

    with right_col:
        st.subheader("Input Summary")
        st.dataframe(input_df, hide_index=True, use_container_width=True)

        st.subheader("Student Performance Analysis")
        if prediction in ["Excellent", "Good"]:
            st.success("The student's current indicators show positive academic performance.")
        elif prediction == "Average":
            st.warning("The student should improve consistency in study hours, assignments, and internal assessment preparation.")
        else:
            st.error("The student needs focused academic mentoring and improvement in key learning indicators.")

        analysis_df = pd.DataFrame(
            {
                "Academic Factor": FEATURE_COLUMNS,
                "Student Value": input_df.iloc[0].values,
                "Dataset Average": df[FEATURE_COLUMNS].mean().round(2).values,
            }
        )
        st.dataframe(analysis_df, hide_index=True, use_container_width=True)

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Graphs and Charts", "Decision Tree", "Dataset Preview", "Model Report"]
    )

    with tab1:
        chart_col1, chart_col2 = st.columns(2, gap="large")
        with chart_col1:
            st.pyplot(plot_performance_distribution(df), use_container_width=True)
        with chart_col2:
            st.pyplot(plot_feature_importance(model), use_container_width=True)

    with tab2:
        st.write("The trained Decision Tree shows how academic factors are split to classify performance.")
        st.pyplot(plot_decision_tree(model), use_container_width=True)

    with tab3:
        st.write("Sample dataset used for model training and testing.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.write("Dataset summary")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab4:
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose().round(3)

        report_col1, report_col2 = st.columns([1, 1], gap="large")
        with report_col1:
            st.pyplot(plot_confusion_matrix(y_test, y_pred), use_container_width=True)
        with report_col2:
            st.write("Classification Report")
            st.dataframe(report_df, use_container_width=True)
            st.write(f"Number of performance categories: {class_count}")
            st.write("Algorithm: Decision Tree Classifier")
            st.write("Train-test split: 75% training and 25% testing")


if __name__ == "__main__":
    main()

