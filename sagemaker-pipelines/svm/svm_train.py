import argparse
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import joblib
import json
import logging

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--gamma', type=str, default='scale')
    parser.add_argument('--class_weight', type=str, default='balanced')
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    return parser.parse_known_args()

def load_data(data_dir):
    input_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    if len(input_files) == 0:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    raw_data = [pd.read_csv(file, header=None) for file in input_files]
    data = pd.concat(raw_data)
    
    y = data.iloc[:, 0].values
    X = data.iloc[:, 1:].values
    
    return X, y

def train_model(args):
    logging.info("Loading training data")
    X_train, y_train = load_data(args.train)
    
    logging.info("Loading validation data")
    X_val, y_val = load_data(args.validation)
    class_weight = args.class_weight
    if class_weight.lower() == 'none':
        class_weight = None
    
    logging.info(f"Training SVM with C={args.C}, kernel={args.kernel}, gamma={args.gamma}, class_weight={class_weight}")
    model = SVC(
        C=args.C,
        kernel=args.kernel,
        gamma=args.gamma,
        class_weight=class_weight,
        probability=True,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on val data
    val_probs = model.predict_proba(X_val)[:, 1]
    val_auc_roc = roc_auc_score(y_val, val_probs)
    
    precision, recall, _ = precision_recall_curve(y_val, val_probs)
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