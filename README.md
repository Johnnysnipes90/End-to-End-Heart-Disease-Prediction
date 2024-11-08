# End-to-End-Heart-Disease-Prediction
End-to-End Heart Disease Prediction Project - a complete pipeline from data preprocessing to model deployment using machine learning techniques. Includes exploratory data analysis, feature engineering, model training, and deployment with FastAPI and Streamlit.

## Project Overview

This project aims to predict the presence of heart disease in patients using a variety of health indicators. The project involves data preprocessing, feature engineering, model training using machine learning algorithms (such as Random Forest and XGBoost), and model evaluation. The project also explores visualizations to understand the relationships between various factors and heart disease outcomes.

## Project Structure

Here’s an overview of the project’s directory structure:

- `assets/imgs/`: Contains images for visualizations or app display.
- `config/`: JSON configuration files for model parameters or other settings.
- `data/`: Contains the heart disease dataset.
- `models/`: Serialized version of the trained models.
- `notebooks/`: Jupyter notebooks for data exploration and prototyping.
- `src/`: Python scripts for data preprocessing and feature engineering.
- `app.py`: Python script for deploying the model as a Flask web application.
- `requirements.txt`: List of dependencies required to run the project.

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

```bash
git clone https://github.com/Johnnysnipes90/End-to-End-Heart-Disease-Prediction.git
cd End-to-End-Heart-Disease-Prediction

## Set Up Virtual Environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

## Install Dependencies
pip install -r requirements.txt

## Run the Streamlit App
## Web UI
![Web UI Example](imgage/web_ui_example.png)



