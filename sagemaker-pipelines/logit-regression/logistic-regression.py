# train_logistic_regression.py
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.inputs import TrainingInput

# Define model output path
model_path = f"s3://{default_bucket}/AbaloneTrain/LogisticRegression"

# Define the SKLearn Estimator
lr_train = SKLearn(
    entry_point="logistic_regression_script.py",  # Your custom training script
    framework_version="1.2-1",
    instance_type=instance_type,
    instance_count=1,
    output_path=model_path,
    role=role,
    sagemaker_session=pipeline_session,
)

# Set hyperparameters
lr_train.set_hyperparameters(
    penalty="l2",
    C=1.0,
    max_iter=100,
)

# Define training inputs
train_args = lr_train.fit(
    inputs={
        "train": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "validation": TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
            content_type="text/csv",
        ),
    }
)


# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_auc_score, precision_score, recall_score
# import joblib
# import os
# import argparse
# from sagemaker.experiments.run import Run
# from sagemaker.session import Session

# def evaluate_model(model, X, y, run):
#     """Evaluate model and log metrics to SageMaker Experiments."""
#     y_pred = model.predict(X)
#     y_pred_proba = model.predict_proba(X)[:, 1]
    
#     metrics = {
#         'auc_roc': roc_auc_score(y, y_pred_proba),
#         'precision': precision_score(y, y_pred),
#         'recall': recall_score(y, y_pred)
#     }
    
#     # Log metrics to SageMaker Experiments
#     for metric_name, metric_value in metrics.items():
#         run.log_metric(metric_name, metric_value)
    
#     return metrics

# def train():
#     parser = argparse.ArgumentParser()
    
#     # SageMaker specific arguments
#     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
#     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
#     parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
#     parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
#     args, _ = parser.parse_known_args()
    
#     # Initialize SageMaker experiment
#     experiment_name = "logistic-regression-training"
#     run = Run.create(experiment_name=experiment_name)
    
#     # Load data
#     X_train = pd.read_parquet(os.path.join(args.train, "X_train.parquet"))
#     y_train = pd.read_parquet(os.path.join(args.train, "y_train.parquet"))
#     X_test = pd.read_parquet(os.path.join(args.test, "X_test.parquet"))
#     y_test = pd.read_parquet(os.path.join(args.test, "y_test.parquet"))
#     X_val = pd.read_parquet(os.path.join(args.val, "X_val.parquet"))
#     y_val = pd.read_parquet(os.path.join(args.val, "y_val.parquet"))
    
#     # Initialize and train model
#     model = LogisticRegression(
#         C=1.0,
#         max_iter=1000,
#         random_state=42
#     )
    
#     # Log hyperparameters
#     run.log_parameters({
#         "C": 1.0,
#         "max_iter": 1000,
#         "random_state": 42
#     })
    
#     # Train model
#     model.fit(X_train, y_train)
    
#     # Evaluate on all sets
#     train_metrics = evaluate_model(model, X_train, y_train, run)
#     val_metrics = evaluate_model(model, X_val, y_val, run)
#     test_metrics = evaluate_model(model, X_test, y_test, run)
    
#     # Log metrics with prefixes
#     for prefix, metrics in [
#         ('train_', train_metrics),
#         ('val_', val_metrics),
#         ('test_', test_metrics)
#     ]:
#         for metric_name, metric_value in metrics.items():
#             run.log_metric(prefix + metric_name, metric_value)
    
#     # Save the model
#     model_path = os.path.join(args.model_dir, 'model.joblib')
#     joblib.dump(model, model_path)
    
#     # End the experiment run
#     run.close()

# if __name__ == '__main__':
#     train()