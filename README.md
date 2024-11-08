# End-to-End-Heart-Disease-Prediction
End-to-End Heart Disease Prediction Project - a complete pipeline from data preprocessing to model deployment using machine learning techniques. Includes exploratory data analysis, feature engineering, model training, and deployment with FastAPI and Streamlit.

## Project Overview

This project aims to predict the presence of heart disease in patients using a variety of health indicators. The project involves data preprocessing, feature engineering, model training using machine learning algorithms (such as Random Forest and XGBoost), and model evaluation. The project also explores visualizations to understand the relationships between various factors and heart disease outcomes.

### Table 1: Heart Disease Data Description

| Variable Name | Description                               | Role   | Type        | Units      |
|---------------|-------------------------------------------|--------|-------------|------------|
| `age`         | Age of the patient                        | Feature| Integer     | years      |
| `sex`         | Gender of the patient                     | Feature| Categorical | -          |
| `cp`          | Chest pain type                           | Feature| Categorical | -          |
| `trestbps`    | Resting blood pressure (on admission)     | Feature| Integer     | mm/Hg      |
| `chol`        | Serum cholesterol                         | Feature| Integer     | mg/dl      |
| `fbs`         | Fasting blood sugar > 120 mg/dl           | Feature| Categorical | -          |
| `restecg`     | Resting electrocardiographic results      | Feature| Categorical | -          |
| `thalach`     | Maximum heart rate achieved               | Feature| Integer     | -          |
| `exang`       | Exercise induced angina                   | Feature| Categorical | -          |
| `oldpeak`     | ST depression induced by exercise relative to rest | Feature | Float | -      |
| `slope`       | Slope of the peak exercise ST segment     | Feature| Categorical | -          |
| `ca`          | Number of major vessels (0-3) colored by fluoroscopy | Feature | Integer | - |
| `thal`        | Thallium stress test result               | Feature| Categorical | -          |
| `status`      | Diagnosis of heart disease                | Target | Categorical | -          |

This dataset will be used to analyze and predict heart disease presence in patients. The models developed in this project aim to support risk analysis and predictive insights in the healthcare sector.

## Project Goals
- Develop machine learning models to predict heart disease risk.
- Analyze feature importance and risk factors associated with heart disease.
- Use data-driven insights to assist healthcare professionals in patient assessment.
## Project Structure

## Project Structure

Here’s an overview of the project structure:

- `config/`: Configuration files for preprocessing and model requirements.
  - `age_group.json`: Defines age group categories.
  - `columns.cp`: Chest pain type mappings.
  - `slope.json`: Slope type mappings.
  - `thal.json`: Thallium stress test mappings.
  - `training_columns.json`: Defines required columns for model training.
- `dataset/`: Contains the datasets used for training and testing.
  - `northern_df.csv`: Dataset for patients from Northern Nigeria.
  - `southern_df.csv`: Dataset for patients from Southern Nigeria.
  - `combined_df.csv`: Combined dataset for comprehensive analysis.
- `image/`: Contains images used in the project.
  - `background.png`: Background image for the Streamlit app.
- `models/`: Folder for saved machine learning models.
  - `best_rf_model.joblib`: Trained Random Forest model for prediction.
- `src/`: Source code for data processing.
  - `__init__.py`: Marks `src` as a package.
  - `preprocess.py`: Script for data preprocessing and feature engineering.
- `venv/`: Virtual environment for dependency management.
- `app.py`: Flask application to provide API endpoints for the model.
- `streamlit.py`: Streamlit app script for the interactive web UI.
- `requirements.txt`: List of dependencies for the project.

## Prerequisites

Before setting up the project, ensure you have the following installed:
- Python 3.10+
- Docker (for containerization, optional)
- Numpy
- Scikit-learn
- Flask (if using for deployment)
- Streamlit (optional for app visualization)

## Setup Instructions

### 1. Clone the Repository

`bash`
git clone https://github.com/Johnnysnipes90/End-to-End-Heart-Disease-Prediction.git
cd End-to-End-Heart-Disease-Prediction

### 2. Set Up Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Run the Flask API
python app.py

### 5. Run the Streamlit App
## Web UI
![Web UI Example](imgage/web_ui_example.png)

## Data Preprocessing
The data preprocessing pipeline is managed by preprocess.py in the src/ folder. Here’s an outline of the preprocessing steps:

Consolidate Datasets: The northern_df and southern_df datasets are merged into combined_df for comprehensive analysis.
Handle Missing Values: Missing values in numerical and categorical columns are imputed based on statistical techniques.
Encoding Categorical Variables: Categorical features like cp (chest pain type), thal, and slope are mapped using the files in the config/ folder.
Feature Scaling: Continuous features such as age and cholesterol levels are standardized.

## Feature Engineering
Age Groups: Patients’ ages are grouped into categories using age_group.json.
Category Mappings: Mappings for cp, slope, and thal are provided in the config files to ensure consistency across datasets.
Selection of Features: The essential features for training, defined in training_columns.json, are selected for the final dataset.

## Model Training
The model pipeline includes the following machine learning algorithms:

Random Forest Classifier: Achieved the best performance and is saved as `best_rf_model.joblib`.
Evaluation: Cross-validation, accuracy, precision, recall, F1-score, and AUC-ROC metrics are used to evaluate model performance.

## Evaluation
Performance Metrics: The model is evaluated using accuracy, precision, recall, F1-score, and AUC-ROC.
Cross-Validation: Cross-validation helps ensure model robustness on unseen data.

## Results
The Random Forest Classifier achieved strong predictive performance with an accuracy of approximately 85%. The model demonstrates solid precision and recall values, making it a reliable choice for predicting heart disease presence in patients.

