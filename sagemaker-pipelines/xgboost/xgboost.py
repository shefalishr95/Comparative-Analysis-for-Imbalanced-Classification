# import json
# import logging
# import pathlib
# import pickle
# import tarfile
# import numpy as np
# import pandas as pd
# import xgboost

# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     confusion_matrix,
#     roc_curve,
#     balanced_accuracy_score
# )

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# logger.addHandler(logging.StreamHandler())


# if __name__ == "__main__":
#     model_path = "/opt/ml/processing/model/model.tar.gz"
#     with tarfile.open(model_path) as tar:
#         tar.extractall(path="..")
    
#     # Load model
#     logger.debug("Loading xgboost model.")
#     model = pickle.load(open("xgboost-model", "rb"))

#     logger.debug("Loading test input data.")
#     test_path = "/opt/ml/processing/test/test.csv"
#     df = pd.read_csv(test_path, header=None)

#     logger.debug("Reading test data.")
#     y_test = df.iloc[:, 0].to_numpy()
#     df.drop(df.columns[0], axis=1, inplace=True)
#     X_test = xgboost.DMatrix(df.values)

#     logger.info("Performing predictions against test data.")
#     prediction_probabilities = model.predict(X_test)
#     predictions = np.round(prediction_probabilities)

#     precision = precision_score(y_test, predictions, zero_division=1)
#     recall = recall_score(y_test, predictions)
#     accuracy = accuracy_score(y_test, predictions)
#     conf_matrix = confusion_matrix(y_test, predictions)
#     fpr, tpr, _ = roc_curve(y_test, prediction_probabilities)
#     balanced_accuracy_score = balanced_accuracy_score(y_test, predictions)

#     logger.debug("Accuracy: {}".format(accuracy))
#     logger.debug("Balanced Accuracy: {}".format(balanced_accuracy_score))
#     logger.debug("Precision: {}".format(precision))
#     logger.debug("Recall: {}".format(recall))
#     logger.debug("Confusion matrix: {}".format(conf_matrix))

#     report_dict = {
#         "binary_classification_metrics": {
#             "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
#             "precision": {"value": precision, "standard_deviation": "NaN"},
#             "recall": {"value": recall, "standard_deviation": "NaN"},
#             "confusion_matrix": {
#                 "0": {"0": int(conf_matrix[0][0]), "1": int(conf_matrix[0][1])},
#                 "1": {"0": int(conf_matrix[1][0]), "1": int(conf_matrix[1][1])},
#             },
#             "receiver_operating_characteristic_curve": {
#                 "false_positive_rates": list(fpr),
#                 "true_positive_rates": list(tpr),
#             },
#             "balanced_accuracy": {"value": balanced_accuracy_score, "standard_deviation": "NaN"},
#         },
#     }

#     output_dir = "/opt/ml/processing/evaluation"
#     pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

#     evaluation_path = f"{output_dir}/evaluation.json"
#     with open(evaluation_path, "w") as f:
#         f.write(json.dumps(report_dict))


# # # train_xgboost.py
# # from sagemaker.estimator import Estimator
# # from sagemaker.inputs import TrainingInput
# # import sagemaker

# # def get_xgboost_training_step_args(default_bucket, region, role, pipeline_session, step_process, instance_type="ml.m5.xlarge"):
# #     model_path = f"s3://{default_bucket}/AbaloneTrain"
# #     image_uri = sagemaker.image_uris.retrieve(
# #         framework="xgboost",
# #         region=region,
# #         version="1.0-1",
# #         py_version="py3",
# #         instance_type=instance_type,
# #     )

# #     xgb_train = Estimator(
# #         image_uri=image_uri,
# #         instance_type=instance_type,
# #         instance_count=1,
# #         output_path=model_path,
# #         role=role,
# #         sagemaker_session=pipeline_session,
# #     )

# #     xgb_train.set_hyperparameters(
# #         objective="reg:linear",
# #         num_round=50,
# #         max_depth=5,
# #         eta=0.2,
# #         gamma=4,
# #         min_child_weight=6,
# #         subsample=0.7,
# #     )

# #     train_args = xgb_train.fit(
# #         inputs={
# #             "train": TrainingInput(
# #                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
# #                 content_type="text/csv",
# #             ),
# #             "validation": TrainingInput(
# #                 s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
# #                 content_type="text/csv",
# #             ),
# #         }
# #     )

# #     return train_args
# #    ),
# #     }
# # )





# # #######################################

# # tuning_job_config = {
# #     "ParameterRanges": {
# #       "CategoricalParameterRanges": [],
# #       "ContinuousParameterRanges": [
# #         {
# #           "MaxValue": "1",
# #           "MinValue": "0",
# #           "Name": "eta"
# #         },
# #         {
# #           "MaxValue": "2",
# #           "MinValue": "0",
# #           "Name": "alpha"
# #         },
# #         {
# #           "MaxValue": "10",
# #           "MinValue": "1",
# #           "Name": "min_child_weight"
# #         }
# #       ],
# #       "IntegerParameterRanges": [
# #         {
# #           "MaxValue": "10",
# #           "MinValue": "1",
# #           "Name": "max_depth"
# #         }
# #       ]
# #     },
# #     "ResourceLimits": {
# #       "MaxNumberOfTrainingJobs": 20,
# #       "MaxParallelTrainingJobs": 3
# #     },
# #     "Strategy": "Bayesian",
# #     "HyperParameterTuningJobObjective": {
# #       "MetricName": "validation:auc",
# #       "Type": "Maximize"
# #     },
# #     "RandomSeed" : 123
# #   }





# # #######################################


# # # import numpy as np
# # # import pandas as pd
# # # import xgboost as xgb
# # # from sklearn.metrics import roc_auc_score, precision_score, recall_score
# # # import joblib
# # # import os
# # # import argparse
# # # from sagemaker.experiments.run import Run
# # # from sagemaker.session import Session

# # # def evaluate_model(model, X, y, run):
# # #     """Evaluate model and log metrics to SageMaker Experiments."""
# # #     y_pred = model.predict(X)
# # #     y_pred_proba = model.predict_proba(X)[:, 1]
    
# # #     metrics = {
# # #         'auc_roc': roc_auc_score(y, y_pred_proba),
# # #         'precision': precision_score(y, y_pred.round()),
# # #         'recall': recall_score(y, y_pred.round())
# # #     }
    
# # #     # Log metrics to SageMaker Experiments
# # #     for metric_name, metric_value in metrics.items():
# # #         run.log_metric(metric_name, metric_value)
    
# # #     return metrics

# # # def train():
# # #     parser = argparse.ArgumentParser()
    
# # #     # SageMaker specific arguments
# # #     parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
# # #     parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
# # #     parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
# # #     parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    
# # #     args, _ = parser.parse_known_args()
    
# # #     # Initialize SageMaker experiment
# # #     experiment_name = "xgboost-training"
# # #     run = Run.create(experiment_name=experiment_name)
    
# # #     # Load data
# # #     X_train = pd.read_parquet(os.path.join(args.train, "X_train.parquet"))
# # #     y_train = pd.read_parquet(os.path.join(args.train, "y_train.parquet"))
# # #     X_test = pd.read_parquet(os.path.join(args.test, "X_test.parquet"))
# # #     y_test = pd.read_parquet(os.path.join(args.test, "y_test.parquet"))
# # #     X_val = pd.read_parquet(os.path.join(args.val, "X_val.parquet"))
# # #     y_val = pd.read_parquet(os.path.join(args.val, "y_val.parquet"))
    
# # #     # Initialize and train model
# # #     model = xgb.XGBClassifier(
# # #         n_estimators=100,
# # #         max_depth=6,
# # #         learning_rate=0.1,
# # #         random_state=42,
# # #         use_label_encoder=False,
# # #         eval_metric='logloss'
# # #     )
    
# # #     # Log hyperparameters
# # #     run.log_parameters({
# # #         "n_estimators": 100,
# # #         "max_depth": 6,
# # #         "learning_rate": 0.1,
# # #         "random_state": 42
# # #     })
    
# # #     # Train model
# # #     model.fit(
# # #         X_train, 
# # #         y_train,
# # #         eval_set=[(X_val, y_val)],
# # #         early_stopping_rounds=10,
# # #         verbose=True
# # #     )
    
# # #     # Evaluate on all sets
# # #     train_metrics = evaluate_model(model, X_train, y_train, run)
# # #     val_metrics = evaluate_model(model, X_val, y_val, run)
# # #     test_metrics = evaluate_model(model, X_test, y_test, run)
    
# # #     # Log metrics with prefixes
# # #     for prefix, metrics in [
# # #         ('train_', train_metrics),
# # #         ('val_', val_metrics),
# # #         ('test_', test_metrics)
# # #     ]:
# # #         for metric_name, metric_value in metrics.items():
# # #             run.log_metric(prefix + metric_name, metric_value)
    
# # #     # Save the model
# # #     model_path = os.path.join(args.model_dir, 'model.joblib')
# # #     joblib.dump(model, model_path)
    
# # #     # End the experiment run
# # #     run.close()

# # # if __name__ == '__main__':
# # #     train()