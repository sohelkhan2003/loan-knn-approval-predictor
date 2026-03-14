# Loan Approval Risk Prediction using Machine Learning (KNN)

## Project Overview

Financial institutions receive thousands of loan applications every day.
Evaluating each application manually is time-consuming and can lead to human bias or incorrect decisions.

Approving a risky loan applicant can result in **loan default**, causing significant financial losses to banks.

This project builds a **Machine Learning based Loan Approval Prediction System** that helps banks automatically determine whether a loan application should be **Approved** or **Rejected** based on an applicant’s financial and demographic information.

The system uses the **K-Nearest Neighbors (KNN)** algorithm and provides predictions through a **Flask-based web application**.

Users can enter applicant details in the web interface and receive **real-time loan approval predictions**.

---

# Business Problem

Banks aim to achieve two critical objectives:

1. **Approve genuine applicants** who are capable of repaying loans.
2. **Reject risky applicants** who are likely to default.

The most costly mistake for banks is called:

**False Approval**

This occurs when a high-risk borrower is mistakenly approved and later fails to repay the loan.

Therefore, the goal of this project is not just high accuracy, but also the ability to **correctly identify high-risk applicants**.

---

# Project Pipeline

The project follows a complete machine learning pipeline:

1. Data Collection
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Engineering
5. Feature Encoding
6. Feature Scaling
7. Model Training
8. Hyperparameter Optimization
9. Model Evaluation
10. Model Deployment using Flask

---

# Dataset Description

The dataset contains historical loan application records.

Each record represents a loan applicant and includes financial and demographic attributes used to determine whether their loan was approved.

The dataset is commonly used for machine learning classification problems.

---

# Target Variable

| Value | Meaning       |
| ----- | ------------- |
| Y     | Loan Approved |
| N     | Loan Rejected |

This is a **binary classification problem** where the model predicts whether a loan will be approved or rejected.

---

# Features Used

| Feature           | Description                            |
| ----------------- | -------------------------------------- |
| Gender            | Gender of the applicant                |
| Married           | Applicant marital status               |
| Dependents        | Number of dependents                   |
| Education         | Graduate or Not Graduate               |
| Self_Employed     | Whether the applicant is self employed |
| ApplicantIncome   | Income of the primary applicant        |
| CoapplicantIncome | Income of co-applicant                 |
| LoanAmount        | Loan amount requested                  |
| Loan_Amount_Term  | Loan repayment period                  |
| Credit_History    | Previous credit repayment history      |
| Property_Area     | Location of the property               |

---

# Exploratory Data Analysis (EDA)

Exploratory Data Analysis was performed to understand patterns in the dataset.

Key insights discovered:

### Loan Approval Distribution

The dataset is **imbalanced**.

* Approved Loans ≈ 68%
* Rejected Loans ≈ 32%

Because of this imbalance, accuracy alone cannot be relied upon for evaluation.

Additional metrics like **Recall, Precision, and F1-score** are required.

---

### Income Distribution

Applicant income shows a **right-skewed distribution**.

This means:

* Most applicants fall within **low to medium income groups**
* A small number of applicants have extremely high incomes

---

### Loan Amount Distribution

Most loan requests fall between **100 and 150 units**, with a few large outliers.

---

### Key Insight

The feature with the strongest influence on loan approval is:

**Credit History**

Applicants with **Credit_History = 1.0** have a significantly higher probability of approval.

This indicates that past repayment behavior strongly predicts future repayment ability.

---

# Data Preprocessing

Before training the model, several preprocessing steps were applied.

---

## Missing Value Treatment

The dataset contained missing values which were handled as follows.

Categorical features filled using **Mode**:

* Gender
* Married
* Dependents
* Self_Employed
* Credit_History

Numerical features filled using **Median**:

* LoanAmount

Median was chosen to reduce the influence of extreme values.

---

# Feature Encoding

Machine learning models require numerical input.

Categorical variables were converted into numeric values.

### Binary Encoding

Applied to features with two categories:

* Gender
* Married
* Self_Employed

---

### Ordinal Encoding

Applied to features with ordered categories:

* Education
* Dependents

---

### One-Hot Encoding

Applied to:

* Property_Area

This avoids incorrect ordering relationships between categories.

---

# Feature Scaling

The KNN algorithm uses **distance calculations** to identify nearest neighbors.

If features have very different scales, the model may become biased toward large-value features.

Example:

ApplicantIncome values are much larger than Dependents values.

Therefore, **feature scaling** is required.

---

## Scaling Methods Tested

| Method         | Accuracy |
| -------------- | -------- |
| No Scaling     | 65%      |
| StandardScaler | 83.7%    |
| MinMaxScaler   | 82.9%    |

**StandardScaler achieved the best performance** and was selected for the final model.

---

# Model Selection

The model used in this project is:

**K-Nearest Neighbors (KNN)**

KNN classifies a data point based on the **majority class of its nearest neighbors**.

---

## Advantages of KNN

* Simple and intuitive
* Works well for structured datasets
* No explicit training phase
* Easy to implement

---

## Limitations of KNN

* Prediction becomes slower for large datasets
* Memory intensive
* Sensitive to feature scaling

---

# Hyperparameter Tuning

The most important parameter in KNN is:

**K — Number of Nearest Neighbors**

Different K values from **1 to 40** were tested.

---

## Optimal Value

The best performing value was:

**K = 2**

This value produced the best balance between accuracy and recall for identifying risky applicants.

---

# Model Performance

| Metric             | Score |
| ------------------ | ----- |
| Accuracy           | 0.73  |
| Precision          | 0.85  |
| Recall (Approval)  | 0.74  |
| Recall (Rejection) | 0.71  |
| F1 Score           | 0.79  |
| ROC AUC            | 0.77  |

---

# Confusion Matrix

[[27 11]
[22 63]]

### Interpretation

27 → Correctly rejected high-risk applicants
63 → Correctly approved safe applicants
11 → False approvals
22 → False rejections

False approvals represent the **highest financial risk for banks**, therefore minimizing them is critical.

---

# Model Deployment

The trained model was deployed using **Flask**.

Flask provides a lightweight web framework to integrate machine learning models into web applications.

The application allows users to:

* Enter loan applicant information
* Submit the form
* Receive instant loan approval predictions

---

# Technologies Used

Python
Scikit-Learn
Flask
Pandas
NumPy
Matplotlib
Seaborn
HTML
CSS

---

## Project Structure

```
loan-knn-approval-predictor/
│
├── app/
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── model/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── notebook/
│   └── knn_model_training.ipynb
│
├── data/
│   └── train.csv
│
├── images/
│   └── eda_visualizations/
│
├── requirements.txt
│
└── README.md
```

### Folder Explanation

**app/**
Contains the Flask web application code and HTML templates used for the user interface.

**model/**
Stores the trained machine learning model and preprocessing files.

* `model.pkl` → Trained KNN model
* `scaler.pkl` → Feature scaling object
* `columns.pkl` → List of model input features

**notebook/**
Contains the Jupyter Notebook used for data analysis, model training, and experimentation.

**data/**
Stores the dataset used for training and testing the model.

**images/**
Contains visualizations generated during Exploratory Data Analysis (EDA).

**requirements.txt**
Lists all Python dependencies required to run the project.

**README.md**
Project documentation explaining the purpose, methodology, and usage of the system.

# How to Run the Project

Clone the repository

git clone https://github.com/sohelkhan2003/loan-knn-approval-predictor.git

Install dependencies

pip install -r requirements.txt

Run the Flask application

cd app
python app.py

Open the browser

http://127.0.0.1:5000

---

# Limitations

The current version has several limitations.

KNN is computationally expensive for large datasets.

Feature importance cannot be easily interpreted.

The dataset size is relatively small.

---

# Future Improvements

Several improvements can be made:

Use advanced models such as Random Forest, XGBoost, and Gradient Boosting.

Implement cross validation.

Add Explainable AI techniques like SHAP or LIME.

Deploy the application on cloud platforms such as AWS, GCP, or Azure.

---

# Author

Sohel Khan
Machine Learning & Data Science Enthusiast
