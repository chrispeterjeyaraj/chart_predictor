import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from chart_type_predictor_data_prep import ChartTypeDataPreparation


class ChartTypeModelTrainer:
    def __init__(self):
        self.data_dir = 'chart_type_prediction_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def train_model(self):
        """
        Train a machine learning model for chart type prediction

        Returns:
            RandomForestClassifier: Trained model
        """
        # Prepare data
        data_prep = ChartTypeDataPreparation()
        X_train, X_test, y_train, y_test = data_prep.prepare_training_data()

        # Initialize and train the model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for Chart Type Prediction')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'confusion_matrix.png'))

        # Save the model
        joblib.dump(model, os.path.join(self.data_dir, 'chart_type_model.joblib'))

        return model

    def predict_chart_type(self, features):
        """
        Predict chart type for given features

        Args:
            features (dict): Features extracted from a dataset

        Returns:
            str: Predicted chart type
        """
        # Load the model and scaler
        model = joblib.load(os.path.join(self.data_dir, 'chart_type_model.joblib'))
        scaler = joblib.load(os.path.join(self.data_dir, 'feature_scaler.joblib'))

        # Convert features to numpy array and scale
        feature_array = np.array(list(features.values())).reshape(1, -1)
        scaled_features = scaler.transform(feature_array)

        # Predict chart type
        prediction = model.predict(scaled_features)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(scaled_features)[0]
        prediction_proba = dict(zip(model.classes_, probabilities))

        return prediction, prediction_proba


# Example usage
if __name__ == '__main__':
    trainer = ChartTypeModelTrainer()

    # Train the model
    model = trainer.train_model()

    # Example prediction
    sample_features = {
        'numeric_columns_count': 2,
        'categorical_columns_count': 1,
        'total_columns': 3,
        'data_point_count': 10,
        'has_time_column': False,
        'max_numeric_range': 50,
        'mean_numeric_values': 25,
        'numeric_variance': 10
    }

    predicted_type, probabilities = trainer.predict_chart_type(sample_features)
    print("Predicted Chart Type:", predicted_type)
    print("Prediction Probabilities:", probabilities)