import argparse
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import joblib
import json
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_samples', type=float, default=1.0)
    parser.add_argument('--contamination', type=float, default=0.1)
    parser.add_argument('--max_features', type=float, default=1.0)
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    return parser.parse_known_args()

def load_data(data_dir):
    logging.info(f"Loading data from {data_dir}")
    input_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    if len(input_files) == 0:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    raw_data = [pd.read_csv(file, header=None) for file in input_files]
    data = pd.concat(raw_data)
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    
    logging.debug(f"Loaded data shape: {data.shape}")
    return X, y

def train_model(args):
    logging.info("Loading training data")
    X_train, y_train = load_data(args.train)
    
    logging.info("Loading validation data")
    X_val, y_val = load_data(args.validation)
    
    # Train Isolation Forest on normal data only 
    X_train_normal = X_train[y_train == 0]
    
    logging.info(f"Training Isolation Forest with n_estimators={args.n_estimators}, "
                 f"max_samples={args.max_samples}, contamination={args.contamination}, "
                 f"max_features={args.max_features}")
    
    model = IsolationForest(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples,
        contamination=args.contamination,
        max_features=args.max_features,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_normal)
    logging.info("Model training completed")

    # lower scores mean more anomalous
    val_scores = -model.decision_function(X_val)  
    val_auc_roc = roc_auc_score(y_val, val_scores)
    precision, recall, _ = precision_recall_curve(y_val, val_scores)
    val_auc_pr = auc(recall, precision)
    
    logging.info(f"Validation ROC AUC: {val_auc_roc:.4f}")
    logging.info(f"Validation PR AUC: {val_auc_pr:.4f}")
    
    # Save metrics and model
    metrics = {
        'validation_auc': float(val_auc_roc),
        'validation_aucpr': float(val_auc_pr)
    }
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")
    metrics_path = os.path.join(args.output_data_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    logging.info(f"Metrics saved to {metrics_path}")
    
    return model, metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args, _ = parse_args()
    train_model(args)