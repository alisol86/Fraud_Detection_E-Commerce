from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, make_scorer, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
import numpy as np
import tensorflow as tf

import pickle

from feature_engineering import FeatureEngineering
from data_loader import DataLoader
from utils import read_config


def split_data(data):
    """Split data into training and test sets"""
    y = data['class']
    X = data.drop(['class'], axis=1)

    # Split into train and holdout test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X, y, X_train, X_test, y_train, y_test

def fit_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Funciton to fit the model, make prediction on the test set and print out evaluation metrics"""
    # Fit the model on the entire training set
    model.fit(X_train, y_train)

    # Predict on the holdout test set
    y_pred = model.predict(X_test)

    print(f"{model_name} Evaluation Metrics: ")
    # Evaluate the model on the holdout test set
    print(classification_report(y_test, y_pred))  # Show precision, recall, F1-score

def cross_val(model, X, y):
    # run cross-validation on the entire dataset
    scoring = {
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1_score': make_scorer(f1_score)
    }
    print(cross_validate(model, X, y, scoring=scoring, cv=5))

def save_model(output_path, model):
    # save model
    with open(output_path,'wb') as f:
        pickle.dump(model,f)

def run_modeling(model, X, y, X_train, y_train, X_test, y_test, model_name, output_path):
    fit_evaluate_model(model, X_train, y_train, X_test, y_test, model_name)
    cross_val(model, X, y)
    save_model(output_path, model)

def sampling_smote(X_train,y_train ):
    """Oversampling minority class using SMOTE method"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def logReg_basic(X, y, X_train, X_test, y_train, y_test):
    # Initialize the logistic regression model
    logistic = LogisticRegression()
    run_modeling(logistic, X, y, X_train, y_train, X_test, y_test, 'Logistic Regression', 'models/logRed_basic.pkl')

def logReg_balanced(X, y, X_train, X_test, y_train, y_test):
    # Initialize the logistic regression model
    logistic = LogisticRegression(class_weight='balanced')
    run_modeling(logistic, X, y, X_train, y_train, X_test, y_test, 'Logistic Regression Balanced', 'models/logRed_balanced.pkl')

def randomForest_basic(X, y, X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(random_state=42)
    run_modeling(rf, X, y, X_train, y_train, X_test, y_test, 'Random Forest', 'models/randomForest_basic.pkl')

def randomForest_balanced(X, y, X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    run_modeling(rf, X, y, X_train, y_train, X_test, y_test, 'Random Forest Balanced', 'models/randomForest_balanced.pkl')

def XGB(X, y, X_train, X_test, y_train, y_test):
    # Initialize the XGBClassifier
    xgb = XGBClassifier(random_state=42, eval_metric='mlogloss')
    run_modeling(xgb, X, y, X_train, y_train, X_test, y_test, 'XGBoost', 'models/XGB.pkl')

def ensembel_classifier(X, y, X_train, X_test, y_train, y_test):
    model1 = RandomForestClassifier(random_state=42)
    model2 = LogisticRegression(random_state=42)
    model3 = XGBClassifier(random_state=42)

    ensemble_model = VotingClassifier(estimators=[('rf', model1), ('lr', model2), ('xgb', model3)], voting='soft')
    run_modeling(ensemble_model, X, y, X_train, y_train, X_test, y_test, 'Ensemble Classifier', 'models/ensembl.pkl')

def neural_network(X_train, X_test, y_train, y_test, output_path):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    n_inputs = X_train.shape[1]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_inputs,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    print(model.summary())

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['Precision', 'Recall'])
    model.fit(X_train, y_train, validation_split=0.2, batch_size=32, epochs=100, shuffle=True, verbose=2)
    y_pred_prob = model.predict(X_test, batch_size=32)  # Predicted probabilities
    y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions (for binary classification)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    save_model(output_path, model)

if __name__ == "__main__":
    # read_data
    data_cfg = read_config('configs/data_loader.yaml')
    fraud_data = DataLoader(data_cfg).read_fraud_data()
    ip_country_data =  DataLoader(data_cfg).read_ip_mapping()

    # calculate features
    features_cfg = read_config('configs/feature_engineering.yaml')
    features = FeatureEngineering(features_cfg, fraud_data, ip_country_data)
    data_features = features()

    # split data
    X, y, X_train, X_test, y_train, y_test = split_data(data_features)

    # train and save different models
    # logistic regression
    logReg_basic(X, y, X_train, X_test, y_train, y_test)
    # logistic regression balanced
    logReg_balanced(X, y, X_train, X_test, y_train, y_test)
    # random forest
    randomForest_basic(X, y, X_train, X_test, y_train, y_test)
    # random forest balanced
    randomForest_balanced(X, y, X_train, X_test, y_train, y_test)
    # XGB
    XGB(X, y, X_train, X_test, y_train, y_test)
    # ensemble model
    ensembel_classifier(X, y, X_train, X_test, y_train, y_test)
    # NN
    neural_network(X_train, X_test, y_train, y_test, 'models/NN.keras')

