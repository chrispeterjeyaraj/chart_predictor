import os
import sys
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the directory containing your modules to the Python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(base_dir)

from chart_type_model_training import ChartTypeModelTrainer
from chart_type_predictor_data_prep import ChartTypeDataPreparation

app = Flask(__name__)
CORS(app)

class ChartTypePredictionService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ChartTypePredictionService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Ensure data and model directories exist
        os.makedirs('chart_type_prediction_data', exist_ok=True)

        # Train model if not already trained
        self.trainer = ChartTypeModelTrainer()
        self.data_prep = ChartTypeDataPreparation()

        # Check if model exists, if not train it
        model_path = os.path.join('chart_type_prediction_data', 'chart_type_model.joblib')
        scaler_path = os.path.join('chart_type_prediction_data', 'feature_scaler.joblib')

        try:
            if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
                logger.info("Model not found. Training model...")
                self.trainer.train_model()

            # Load model and scaler
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Model and scaler loaded successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def extract_features(self, data):
        """
        Extract features from input data

        Args:
            data (list): List of dictionaries representing collection data

        Returns:
            dict: Extracted features
        """
        try:
            features = self.data_prep.extract_features(data)
            logger.info(f"Features extracted: {features}")
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise

    def predict_chart_type(self, features):
        """
        Predict chart type for given features

        Args:
            features (dict): Features extracted from a dataset

        Returns:
            dict: Prediction results
        """
        try:
            # Convert features to numpy array and scale
            feature_array = np.array(list(features.values())).reshape(1, -1)
            scaled_features = self.scaler.transform(feature_array)

            # Predict chart type
            prediction = self.model.predict(scaled_features)[0]

            # Get prediction probabilities
            probabilities = self.model.predict_proba(scaled_features)[0]
            prediction_proba = dict(zip(self.model.classes_, probabilities))

            logger.info(f"Prediction: {prediction}, Probabilities: {prediction_proba}")

            return {
                'predicted_chart_type': prediction,
                'prediction_probabilities': prediction_proba
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise


# Initialize prediction service
try:
    prediction_service = ChartTypePredictionService()
except Exception as e:
    logger.critical(f"Failed to initialize prediction service: {e}")
    # This will prevent the app from starting if model initialization fails
    raise


def convert_numpy_to_native(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(v) for v in obj]
    return obj


@app.route('/api/predict_chart_type', methods=['POST'])
def predict_chart_type():
    try:
        # Get data from request
        data = request.json.get('data', [])
        logger.info(f"Received prediction request with data: {data}")

        # Validate input
        if not data or not isinstance(data, list):
            logger.warning("Invalid input received")
            return jsonify({
                'error': 'Invalid input. Expected a list of dictionaries.',
                'status': 'error'
            }), 400

        # Extract features
        features = prediction_service.extract_features(data)

        # Predict chart type
        prediction = prediction_service.predict_chart_type(features)

        # Recursively convert NumPy types to native Python types
        converted_prediction = convert_numpy_to_native(prediction)
        converted_features = convert_numpy_to_native(features)

        # Add extracted features to response
        converted_prediction['extracted_features'] = converted_features

        logger.info(f"Prediction response: {converted_prediction}")
        return jsonify(converted_prediction)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        # Additional health checks can be added here
        return jsonify({
            'status': 'healthy',
            'message': 'Chart Type Prediction Service is running',
            'model_loaded': hasattr(prediction_service, 'model')
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)