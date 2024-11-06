import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset paths
DATA_PATH_North = os.getenv("DATA_PATH_NORTH", default="northern heart disease data.csv")
DATA_PATH_South = os.getenv("DATA_PATH_SOUTH", default="southern heart disease data.csv")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", default="combined_df.csv")

class Preprocessor:
    def __init__(self):
        # Load and combine datasets
        self.df_north = pd.read_csv(DATA_PATH_North)
        self.df_south = pd.read_csv(DATA_PATH_South)
        self.df_combined = pd.concat([self.df_north, self.df_south], axis=0)
        self.clean_data()
        self.impute_missing_values()
        self.convert_data_types()
        self.log_transform_features()
        self.categorize_age()
        self.map_binary_variables()
        self.one_hot_encode()
        self.split_and_scale_data()
    
    def clean_data(self):
        # Drop unnecessary columns
        self.df_combined.drop(columns=['Unnamed: 14', 'Unnamed: 15'], inplace=True)

    def impute_missing_values(self):
        # Fill missing values with median or mode
        self.df_combined['age'].fillna(self.df_combined['age'].median(), inplace=True)
        self.df_combined['trestbps'].fillna(self.df_combined['trestbps'].median(), inplace=True)
        self.df_combined['thalach'].fillna(self.df_combined['thalach'].median(), inplace=True)
        self.df_combined['ca'].fillna(self.df_combined['ca'].mode()[0], inplace=True)
        self.df_combined['thal'].fillna(self.df_combined['thal'].mode()[0], inplace=True)

    def convert_data_types(self):
        # Convert columns to appropriate data types
        self.df_combined['age'] = self.df_combined['age'].astype('int')
        self.df_combined['trestbps'] = self.df_combined['trestbps'].astype('int')
        self.df_combined['restecg'] = self.df_combined['restecg'].astype(str)
        self.df_combined['thalach'] = self.df_combined['thalach'].astype('int')
        self.df_combined['ca'] = self.df_combined['ca'].astype('int')

    def log_transform_features(self):
        # Apply log transformation to specific columns
        self.df_combined['trestbps'] = np.log(self.df_combined['trestbps'] + 1)
        self.df_combined['oldpeak'] = np.log(self.df_combined['oldpeak'] + 1)
        self.df_combined['chol'] = np.log(self.df_combined['chol'] + 1)

    def categorize_age(self):
        # Categorize age into groups
        bins = [0, 12, 20, 40, 60, 80, float('inf')]
        labels = ['Child', 'Teen', 'Adult', 'Middle-aged', 'Senior', "Over-aged"]
        self.df_combined['age_group'] = pd.cut(self.df_combined['age'], bins=bins, labels=labels, right=False)

    def map_binary_variables(self):
        # Map binary categorical variables to numerical format
        self.df_combined['sex'] = self.df_combined['sex'].map({'male': 1, 'female': 0})
        self.df_combined['exang'] = self.df_combined['exang'].map({'yes': 1, 'no': 0})
        self.df_combined['status'] = self.df_combined['status'].map({'present': 1, 'absent': 0})
        self.df_combined['fbs'] = self.df_combined['fbs'].map({'yes': 1, 'no': 0})

    def one_hot_encode(self):
        # One-Hot Encoding for multi-class categorical variables
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
        # Separate features and target variable
        X = self.df_combined.drop('status', axis=1)
        y = self.df_combined['status']
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save preprocessed data (optional)
        self.X_train_scaled, self.X_test_scaled = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test

    def save_combined_data(self):
        # Save the combined dataframe as CSV
        self.df_combined.to_csv(OUTPUT_PATH, index=False)

# Example usage:
# preprocessor = Preprocessor()
# preprocessor.save_combined_data()
# Now, preprocessor.X_train_scaled, preprocessor.X_test_scaled, preprocessor.y_train, and preprocessor.y_test
# contain the processed and split data.
