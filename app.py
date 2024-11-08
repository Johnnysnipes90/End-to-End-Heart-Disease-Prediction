import json
import os
import logging
from flask import Flask, request, jsonify
import pandas as pd
from waitress import serve
from src.preprocess import Preprocessor
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)

MODEL_PATH = os.getenv("MODEL_PATH", default="model/best_rf_model.joblib")

# Load the model
try:
    loaded_model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

app = Flask(__name__)

# Define a home route that returns a simple JSON response
@app.route("/")
def home():
    return jsonify({"message": "Hello, Heart!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the JSON payload from the request
        payload = request.get_json()

        # Ensure the payload is not empty or invalid
        if not payload:
            raise ValueError("No input data provided")

        # Convert the JSON payload to a pandas DataFrame
        data = pd.DataFrame(payload)

        # Check if the dataframe is empty or malformed
        if data.empty:
            raise ValueError("Received empty data")

        # Preprocess the data using the updated Preprocessor class
        preprocessor = Preprocessor()
        processed_data = preprocessor.preprocess()

        # Make prediction using the loaded model
        prediction = loaded_model.predict(processed_data)

        # Return the prediction as a JSON response
        return jsonify(prediction.tolist())

    except Exception as e:
        # Log the error and return a detailed message
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    serve(app, host='127.0.0.1', port=8000)