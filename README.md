# **End-to-End-Ml-HR-Attrition-ML_Project**

## **Project Overview**

This project focuses on predicting employee attrition using machine learning techniques and creating an interactive Tableau dashboard for exploratory data analysis (EDA). We will use the **IBM HR Analytics Employee Attrition** dataset to analyze various factors influencing employee turnover and develop models to predict whether an employee is likely to leave the company.

### **Key Steps in the Project:**
1. **Exploratory Data Analysis (EDA)** using Tableau
2. **Machine Learning Model Building** (Multiple Algorithms)
3. **Model Tracking and Experimentation** with MLflow
4. **Deployment** of the model using Flask

---

## **1. Exploratory Data Analysis (EDA)**

To understand the data and its structure, EDA will be performed using **Tableau**. We will visualize key factors such as:
- Job Satisfaction
- Monthly Income
- Overtime Work
- Job Role
- Age, etc.

### **Steps:**
- Load the HR Attrition dataset into Tableau.
- Create visualizations to analyze patterns between different features and attrition.
- Key visualizations include:
  - Distribution of employees by age, department, and salary.
  - Correlation between features like job satisfaction, overtime, and attrition.
  - Heatmaps for exploring feature correlations.

### **Files:**
- `HR_attrition_data.csv`: Dataset containing employee information.
- Tableau workbook (`HR_Attrition_EDA.twb`): Contains the Tableau dashboard.

---

## **2. Machine Learning Model Building**

Once we have insights from the EDA, the next step is to build and evaluate various machine learning models to predict employee attrition.

### **Steps:**
1. **Preprocessing**:
   - Handle missing values, categorical encoding, and normalization of data.
   - Split the data into training and testing sets.
   
2. **Modeling**:
   Algorithms that will be used are:
   - **Logistic Regression**
   - **Random Forest**
   - **Support Vector Machine (SVM)**
   - **K-Nearest Neighbours
   

3. **Evaluation**:
   - Models will be evaluated based on accuracy, precision, recall, F1-score, and ROC-AUC.
   - Cross-validation and hyperparameter tuning will be applied to improve model performance.

4. **Explainability**:
   - Use SHAP or LIME to explain the predictions made by the models.

### **Files:**
- `model_training.ipynb`: Jupyter notebook with data preprocessing, model training, and evaluation.

---

## **3. Experiment Tracking with MLflow**

To efficiently manage multiple models and hyperparameters, we will use **MLflow** for tracking experiments.

### **Steps:**
1. **Set up MLflow**:
   - Initialize an MLflow server to log the results of different models and hyperparameters.
   - Log metrics such as accuracy, precision, recall, and F1-score.

2. **Track Multiple Models**:
   - Compare models based on their performance using MLflow’s UI.

# Start MLflow UI
mlflow ui
## **Logs to be Tracked:**
- Model parameters and metrics.
- Model artifacts (trained models).

### **Files:**
- `mlflow_tracking.py`: Python script for training models and logging results to MLflow.

---

## **4. Model Deployment with Flask**

The final step of the project is to deploy the best-performing machine learning model using **Flask**. The model will be served via a web API that can be used to predict whether a new employee is at risk of leaving.

### **Steps:**

1. **Build a Flask API:**
   - Create endpoints for loading the trained model and making predictions.
   - The API will take employee details as input and return the prediction (attrition or no attrition).

2. **Deploy the Flask API:**
   - Deploy locally or on a cloud platform (e.g., Heroku, AWS).

3. **Usage:**
   - Send a POST request with employee data in JSON format.
   - Get the attrition prediction in the response.

### **Files:**
- `app.py`: Flask app that serves the prediction model.
- `model.pkl`: Serialized best-performing model.
- `requirements.txt`: Required Python packages.

---

## **Project Structure**
``` bash
.
├── HR_attrition_data.csv          # Dataset
├── HR_Attrition_EDA.twb           # Tableau workbook for EDA
├── model_training.ipynb           # Jupyter notebook for model training
├── mlflow_tracking.py             # Python script for MLflow tracking
├── app.py                         # Flask app for model deployment
├── model.pkl                      # Trained and saved model
├── requirements.txt               # Python dependencies
├── Questions for visualtization   # Text File            
└── README.md                      # Project overview

```
## **How to Run the Project**

### **1. Clone the Repository:**
```bash
git clone https://github.com/your-username/employee-attrition-prediction.git
cd employee-attrition-prediction
```
Run the MLflow Server:
```bash
mlflow ui
```

Run the Flask App:
``` bash

python app.py
```
The Flask API will be available at http://127.0.0.1:5000.
