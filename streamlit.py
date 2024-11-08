import os
import json
import requests
import base64

import pandas as pd
import streamlit as st

from src.preprocess import Preprocessor  # Ensure this points to your preprocessing script

# Define data paths for heart disease prediction
DATA_PATH = os.getenv("DATA_PATH", default="dataset/combined_df.csv")
COLUMN_CONFIG = os.getenv("COLUMN_CONFIG", default="config/columns.json")
AGE_GROUP_CONFIG = os.getenv("AGE_GROUP_CONFIG", default="config/age_group.json")
CP_CONFIG = os.getenv("CP_CONFIG", default="config/cp.json")
SLOPE_CONFIG = os.getenv("SLOPE_CONFIG", default="config/slope.json")
THAL_CONFIG = os.getenv("THAL_CONFIG", default="config/thal.json")

BG_IMAGE_PATH = "image/heart img.png"

# Set the background image for the app
@st.cache_resource
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file, opacity=0.2):  # Increased opacity for better contrast
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})), url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_png_as_page_bg(BG_IMAGE_PATH, opacity=0.2)

# Custom CSS for styling the text
st.markdown("""
    <style>
        .css-1v3fvcr { font-size: 24px; }  /* Increase title size */
        .css-18e3th9 { font-size: 18px; }  /* Increase sidebar labels size */
        h2 { font-size: 30px; font-weight: bold; }
        .stButton>button { font-size: 18px; padding: 10px; }  /* Increase button size */
    </style>
""", unsafe_allow_html=True)

st.title("Heart Disease Prediction")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)
data = load_data()

@st.cache_data
def load_columns():
    with open(COLUMN_CONFIG, "r") as f:
        return json.load(f)
columns = load_columns()

@st.cache_data
def load_age_group():
    with open(AGE_GROUP_CONFIG, "r") as f:
        return json.load(f)
age_group = load_age_group()

@st.cache_data
def load_cp():
    with open(CP_CONFIG, "r") as f:
        return json.load(f)
cp = load_cp()

@st.cache_data
def load_slope():
    with open(SLOPE_CONFIG, "r") as f:
        return json.load(f)
slope = load_slope()

@st.cache_data
def load_thal():
    with open(THAL_CONFIG, "r") as f:
        return json.load(f)
thal = load_thal()

with st.sidebar:
    st.write("### Patient Information") 

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0)
        trestbps = st.number_input("Resting Blood Pressure", min_value=0)
        chol = st.number_input("Cholesterol", min_value=0)
    with col2:
        sex = st.selectbox("Sex", data["sex"].dropna().unique())
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0)
    
    exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
    cp_type = st.selectbox("Chest Pain Type", cp.keys())
    restecg = st.selectbox("Resting ECG Results", data["restecg"].dropna().unique())
    slope_value = st.selectbox("Slope of ST Segment", slope.keys())
    thal_value = st.selectbox("Thalassemia", thal.keys())

    payload = {
        "age": [age],
        "sex": [sex],
        "cp": [cp_type],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [1 if fbs == "Yes" else 0],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [1 if exang == "Yes" else 0],
        "slope": [slope_value],
        "thal": [thal_value],
    }

    # Convert the input to a DataFrame
    df = pd.DataFrame(payload)

st.write("### Patient Information")
st.write(df)

st.write("### JSON Payload")
st.json(payload, expanded=False)

if st.button("Predict", key="predict", help="Click to make a prediction"):
    with st.spinner("Making Prediction... Please wait"):
        response = requests.post("http://127.0.0.1:8000/predict", json=df.to_dict())

        if response.status_code == 200:
            prediction = response.json()[0]
            if prediction == 1:
                st.write("### Prediction")
                st.markdown(f"<h2 style='color:blue;'>Heart Disease Detected</h2>", unsafe_allow_html=True)
            else:
                st.write("### Prediction")
                st.markdown(f"<h2 style='color:green;'>No Heart Disease Detected</h2>", unsafe_allow_html=True)
            st.success("Prediction made successfully!")
            st.balloons()
        else:
            st.error("Prediction failed! Please check the input data or try again later.")

# Footer with improved style
def footer():
    st.markdown("""
    <hr style="border-top: 1px solid #bbb;"/>
    <p style="text-align: center;">
        <span style="font-weight: bold;">Background image credits:</span> <a href="https://pixabay.com" style="color:#4682B4;">Pixabay</a> | Designed by 
        <a href="https://linkedin.com/in/john-olalemi/" style="color:#4682B4; font-weight: bold;">John Olalemi</a>
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    footer()
