# AI-Based Student Performance Prediction System Using Decision Tree Algorithm

This is a complete MCA Artificial Intelligence mini project that predicts student academic performance using a real Machine Learning workflow and the Decision Tree Classification Algorithm. The system is built with Python, Streamlit, Pandas, NumPy, Scikit-Learn, and Matplotlib.

## Project Objective

The objective of this project is to analyze student academic factors and classify performance into four categories: Excellent, Good, Average, and Poor. The model is trained using attendance percentage, internal assessment marks, assignment score, study hours per day, and previous semester percentage.

## Folder Structure

```text
student-performance-prediction/
├── app.py
├── dataset.csv
├── model_training.py
├── requirements.txt
└── README.md
```

## Features

- Professional Streamlit dashboard
- Student performance prediction panel
- Decision Tree Classifier training
- Accuracy display
- Decision Tree visualization
- Feature importance graph
- Dataset preview
- Student performance distribution chart
- Confusion matrix
- Classification report
- Student performance analysis based on input values

## User Inputs

- Attendance Percentage
- Internal Assessment Marks
- Assignment Score
- Study Hours Per Day
- Previous Semester Percentage

## Output Classes

- Excellent
- Good
- Average
- Poor

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib

## Installation Steps

1. Install Python 3.9 or above.
2. Open a terminal or command prompt.
3. Move into the project folder.

```bash
cd student-performance-prediction
```

4. Install the required libraries.

```bash
pip install -r requirements.txt
```

## Execution Steps

Run the Streamlit application with this command:

```bash
streamlit run app.py
```

After running the command, the application opens in the browser. Enter the student details in the sidebar and click Predict Performance to generate the AI prediction.

## Model Training

The application automatically trains a Decision Tree Classifier when it starts. To train and test the model separately from the command line, run:

```bash
python model_training.py
```

This prints the model accuracy, classification report, feature importance values, and a sample prediction.

## Machine Learning Approach

The dataset is loaded from `dataset.csv`. The features are separated from the target column named `Performance`. The data is split into training and testing sets using a 75:25 ratio. A Decision Tree Classifier is trained on the training data and evaluated on the test data using accuracy, confusion matrix, and classification report.

## Dataset Description

The project includes a sample dataset with 120 records. Each record contains numeric academic factors and one target performance label. The target labels are generated using meaningful academic rules so the Decision Tree can learn realistic classification patterns.

| Column | Description |
|---|---|
| Attendance_Percentage | Student attendance percentage out of 100 |
| Internal_Assessment_Marks | Internal marks out of 50 |
| Assignment_Score | Assignment score out of 100 |
| Study_Hours_Per_Day | Average study hours per day |
| Previous_Semester_Percentage | Previous semester academic percentage |
| Performance | Final class label: Excellent, Good, Average, or Poor |

## MCA Submission Notes

This project genuinely uses Machine Learning because it trains a supervised classification model using labeled data. The Decision Tree algorithm learns decision rules from the dataset and predicts the performance category for new student input. The dashboard also visualizes model accuracy, feature importance, decision tree structure, and dataset analysis.
