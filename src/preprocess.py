import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load dataset path for combined data
DATA_PATH = os.getenv("DATA_PATH", default="dataset/combined_df.csv")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", default="dataset/processed_combined_df.csv")

class Preprocessor:
    def __init__(self):
        logging.info("Initializing Preprocessor...")
        self.df_combined = pd.read_csv(DATA_PATH)
        logging.info("Dataset loaded successfully.")

    def preprocess(self):
        self.clean_data()
        self.impute_missing_values()
        self.convert_data_types()
        self.log_transform_features()
        self.categorize_age()
        self.map_binary_variables()
        self.one_hot_encode()
        self.split_and_scale_data()
        return self.X_test_scaled

    def clean_data(self):
        logging.info("Cleaning data...")
        # Drop any unnecessary columns if they exist
        extra_columns = [col for col in ['Unnamed: 14', 'Unnamed: 15'] if col in self.df_combined.columns]
        self.df_combined.drop(columns=extra_columns, inplace=True)

    def impute_missing_values(self):
        logging.info("Imputing missing values...")
        self.df_combined['age'] = self.df_combined['age'].fillna(self.df_combined['age'].median())
        self.df_combined['trestbps'] = self.df_combined['trestbps'].fillna(self.df_combined['trestbps'].median())
        self.df_combined['thalach'] = self.df_combined['thalach'].fillna(self.df_combined['thalach'].median())
        self.df_combined['ca'] = self.df_combined['ca'].fillna(self.df_combined['ca'].mode()[0])
        self.df_combined['thal'] = self.df_combined['thal'].fillna(self.df_combined['thal'].mode()[0])


    def convert_data_types(self):
        logging.info("Converting data types...")
        self.df_combined['age'] = self.df_combined['age'].astype('int')
        self.df_combined['trestbps'] = self.df_combined['trestbps'].astype('int')
        self.df_combined['restecg'] = self.df_combined['restecg'].astype(str)
        self.df_combined['thalach'] = self.df_combined['thalach'].astype('int')
        self.df_combined['ca'] = self.df_combined['ca'].astype('int')

    def log_transform_features(self):
        logging.info("Applying log transformation to features...")
        self.df_combined['trestbps'] = np.log(self.df_combined['trestbps'] + 1)
        self.df_combined['oldpeak'] = np.log(self.df_combined['oldpeak'] + 1)
        self.df_combined['chol'] = np.log(self.df_combined['chol'] + 1)

    def categorize_age(self):
        logging.info("Categorizing age...")
        bins = [0, 12, 20, 40, 60, 80, float('inf')]
        labels = ['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior', "Over-aged"]
        self.df_combined['age_group'] = pd.cut(self.df_combined['age'], bins=bins, labels=labels, right=False)

    def map_binary_variables(self):
        logging.info("Mapping binary variables...")
        self.df_combined['sex'] = self.df_combined['sex'].map({'male': 1, 'female': 0})
        self.df_combined['exang'] = self.df_combined['exang'].map({'yes': 1, 'no': 0})
        self.df_combined['status'] = self.df_combined['status'].map({'present': 1, 'absent': 0})
        self.df_combined['fbs'] = self.df_combined['fbs'].map({'yes': 1, 'no': 0})

    def one_hot_encode(self):
        logging.info("One-hot encoding categorical variables...")
        cp_dummies = pd.get_dummies(self.df_combined['cp'], prefix='cp')
        slope_dummies = pd.get_dummies(self.df_combined['slope'], prefix='slope')
        thal_dummies = pd.get_dummies(self.df_combined['thal'], prefix='thal')
        age_group_dummies = pd.get_dummies(self.df_combined['age_group'], prefix='age_group')
        
        # Concatenate new dummy variables and drop original columns
        self.df_combined = pd.concat(
            [self.df_combined.drop(['cp', 'slope', 'thal', 'age_group'], axis=1),
             cp_dummies, slope_dummies, thal_dummies, age_group_dummies],
            axis=1
        )

    def split_and_scale_data(self):
        logging.info("Splitting and scaling data...")
        X = self.df_combined.drop('status', axis=1)
        y = self.df_combined['status']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)
        logging.info("Data scaling completed")


    def save_data(self):
        logging.info("Saving processed data...")
        if not os.path.exists(os.path.dirname(OUTPUT_PATH)):
            os.makedirs(os.path.dirname(OUTPUT_PATH))
        self.df_combined.to_csv(OUTPUT_PATH, index=False)
        logging.info(f"Processed data saved to {OUTPUT_PATH}")
