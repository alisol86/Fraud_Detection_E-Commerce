import os
import numpy as np
import pickle
from feature_engineering import FeatureEngineering
from data_loader import DataLoader
from utils import read_config
from sklearn.metrics import classification_report
import tensorflow as tf


def read_test_data():
    # read_data
    data_cfg = read_config('configs/data_loader.yaml')
    fraud_data = DataLoader(data_cfg).read_fraud_data()
    ip_country_data =  DataLoader(data_cfg).read_ip_mapping()

    # calculate features
    features_cfg = read_config('configs/feature_engineering.yaml')
    features = FeatureEngineering(features_cfg, fraud_data, ip_country_data)
    data_features = features()

    y = data_features['class']
    X = data_features.drop(['class'], axis=1)
    return X, y

def inference(model_file, X, y):
    # Load the model
    if model_file.endswith('.pkl'):
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        # Predict test set
        y_pred = model.predict(X)
    elif model_file.endswith('.keras'):
        model = tf.keras.models.load_model(model_file)
        y_pred_prob = model.predict(X, batch_size=32)  # Predicted probabilities
        y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions (for binary classification)

    print(f"{model_file} Evaluation Metrics: ")
    # # Evaluate the model on the holdout test set
    print(classification_report(y, y_pred))  # Show precision, recall, F1-score
    print("######################################")

if __name__ == "__main__":
    X, y = read_test_data()
    model_files = os.listdir('models')
    for model_file in model_files:
        inference("models/"+ model_file, X, y)


