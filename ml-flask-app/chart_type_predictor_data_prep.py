import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os


class ChartTypeDataPreparation:
    def __init__(self):
        self.data_dir = 'chart_type_prediction_data'
        os.makedirs(self.data_dir, exist_ok=True)

    def extract_features(self, data):
        """
        Extract features from a dataset for chart type prediction

        Args:
            data (list or pd.DataFrame): Input dataset

        Returns:
            dict: Extracted features
        """
        # Convert to DataFrame if it's a list of dictionaries
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        # Feature extraction
        features = {
            'numeric_columns_count': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns_count': len(df.select_dtypes(include=['object', 'category']).columns),
            'total_columns': len(df.columns),
            'data_point_count': len(df),
            'has_time_column': any('date' in col.lower() or 'time' in col.lower() for col in df.columns),
            'max_numeric_range': self._calculate_numeric_range(df),
            'mean_numeric_values': self._calculate_mean_numeric(df),
            'numeric_variance': self._calculate_numeric_variance(df)
        }

        return features

    def _calculate_numeric_range(self, df):
        """Calculate the range of numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].max().max() - df[numeric_cols].min().min() if len(numeric_cols) > 0 else 0

    def _calculate_mean_numeric(self, df):
        """Calculate mean of numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].mean().mean() if len(numeric_cols) > 0 else 0

    def _calculate_numeric_variance(self, df):
        """Calculate variance of numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return df[numeric_cols].var().mean() if len(numeric_cols) > 0 else 0

    def generate_synthetic_training_data(self, num_samples=1000):
        """
        Generate synthetic training data for chart type prediction

        Returns:
            pd.DataFrame: Synthetic training dataset
        """
        training_data = []
        chart_types = ['bar', 'line', 'pie', 'scatter']

        for _ in range(num_samples):
            # Randomly generate datasets with different characteristics
            chart_type = np.random.choice(chart_types)

            if chart_type == 'bar':
                length = np.random.randint(3, 10)
                data = pd.DataFrame({
                    'category': [f'Cat_{i}' for i in range(length)],
                    'value': np.random.randint(10, 100, size=length)
                })
            elif chart_type == 'line':
                length = np.random.randint(5, 20)
                data = pd.DataFrame({
                    'date': pd.date_range(start='1/1/2023', periods=length),
                    'value': np.cumsum(np.random.normal(0, 1, length))
                })
            elif chart_type == 'pie':
                length = np.random.randint(3, 6)
                data = pd.DataFrame({
                    'category': [f'Cat_{i}' for i in range(length)],
                    'percentage': np.random.dirichlet(np.ones(length)) * 100
                })
            else:  # scatter
                length = np.random.randint(10, 50)
                data = pd.DataFrame({
                    'x': np.random.normal(0, 1, length),
                    'y': np.random.normal(0, 1, length)
                })

            features = self.extract_features(data)
            features['chart_type'] = chart_type
            training_data.append(features)

        return pd.DataFrame(training_data)

    def prepare_training_data(self):
        """
        Prepare and save training data

        Returns:
            tuple: Training and testing datasets
        """
        # Generate synthetic data
        df = self.generate_synthetic_training_data()

        # Separate features and target
        X = df.drop('chart_type', axis=1)
        y = df['chart_type']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler and datasets
        import joblib
        joblib.dump(scaler, os.path.join(self.data_dir, 'feature_scaler.joblib'))

        return X_train_scaled, X_test_scaled, y_train, y_test


# Example usage
if __name__ == '__main__':
    data_prep = ChartTypeDataPreparation()
    X_train, X_test, y_train, y_test = data_prep.prepare_training_data()
    print("Training data prepared successfully!")