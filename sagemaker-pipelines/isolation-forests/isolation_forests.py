import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import joblib
import os
import argparse
from sagemaker.experiments.run import Run
from sagemaker.session import Session

def evaluate_model(model, X, y, run, threshold=-0.5):
    """
    Evaluate anomaly detection model and log metrics to SageMaker Experiments.
    Note: For anomaly detection, y should be 1 for normal and -1 for anomalies,
    or 0 for normal and 1 for anomalies.
    """
    # Get anomaly scores (-1 for anomalies, 1 for normal points)
    y_pred = model.predict(X)
    # Get decision function scores (more negative = more anomalous)
    decision_scores = model.score_samples(X)
    
    # Convert predictions to binary format if needed (assuming y uses 0/1 format)
    y_binary = y if set(y).issubset({0, 1}) else (y == 1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)
    
    metrics = {
        'auc_roc': roc_auc_score(y_binary, -decision_scores),  # Negative scores because more negative = more anomalous
        'precision': precision_score(y_binary, y_pred_binary),
        'recall': recall_score(y_binary, y_pred_binary)
    }
    
    # Log metrics to SageMaker Experiments
    for metric_name, metric_value in metrics.items():
        run.log_metric(metric_name, metric_value)
    
    return metrics

def train():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
    args, _ = parser.parse_known_args()
    
    # Initialize SageMaker experiment
    experiment_name = "isolation-forest-training"
    run = Run.create(experiment_name=experiment_name)
    
    # Load data
    X_train = pd.read_parquet(os.path.join(args.train, "X_train.parquet"))
    y_train = pd.read_parquet(os.path.join(args.train, "y_train.parquet"))
    X_test = pd.read_parquet(os.path.join(args.test, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(args.test, "y_test.parquet"))
    X_val = pd.read_parquet(os.path.join(args.val, "X_val.parquet"))
    y_val = pd.read_parquet(os.path.join(args.val, "y_val.parquet"))
    
    # Calculate contamination from training data
    contamination = np.mean(y_train == 1)  # Assuming 1 is anomaly
    
    # Initialize and train model
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        max_samples='auto'
    )
    
    # Log hyperparameters
    run.log_parameters({
        "n_estimators": 100,
        "contamination": contamination,
        "random_state": 42,
        "max_samples": 'auto'
    })
    
    # Train model (only on normal samples from training set)
    model.fit(X_train)
    
    # Evaluate on all sets
    train_metrics = evaluate_model(model, X_train, y_train, run)
    val_metrics = evaluate_model(model, X_val, y_val, run)
    test_metrics = evaluate_model(model, X_test, y_test, run)
    
    # Log metrics with prefixes
    for prefix, metrics in [
        ('train_', train_metrics),
        ('val_', val_metrics),
        ('test_', test_metrics)
    ]:
        for metric_name, metric_value in metrics.items():
            run.log_metric(prefix + metric_name, metric_value)
    
    # Save the model
    model_path = os.path.join(args.model_dir, 'model.joblib')
    joblib.dump(model, model_path)
    
    # End the experiment run
    run.close()

if __name__ == '__main__':
    train()