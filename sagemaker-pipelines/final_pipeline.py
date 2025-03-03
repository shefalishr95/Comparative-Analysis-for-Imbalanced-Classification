import boto3
import json
import sagemaker
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, CacheConfig, TuningStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.workflow.steps import TuningStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.properties import PropertyFile
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# SageMaker session setup
session = sagemaker.session.Session()
region = session.boto_region_name
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession()
default_bucket = "amazon-product-dataset-2024"
prefix = "model-comparison"

def get_pipeline(pipeline_name):
    logger.info("Initializing pipeline parameters")
    
    # Pipeline params
    input_data = ParameterString(
        name="InputData", 
        default_value=f"s3://{default_bucket}/transformed/transformed-data.parquet"
    )
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.2xlarge")
    max_jobs = ParameterInteger(name="MaxJobs", default_value=5)
    max_parallel_jobs = ParameterInteger(name="MaxParallelJobs", default_value=2)
    
    # Cache config
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    
    # Step 1. Data preprocessing 
    logger.info("Creating preprocessing step")
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="data-preprocessing",
        role=role,
        sagemaker_session=pipeline_session
    )

    step_process = ProcessingStep(
        name="Prepare-Data",
        display_name="Preprocessing",
        processor=sklearn_processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
        ],
        code="s3://amazon-product-dataset-2024/scripts/preprocessing.py",
        cache_config=cache_config
    )
    
    model_steps = {} # store model training steps
    evaluation_steps = {}
    
    # Step 2: Create steps for each model (to run concurrently)
    ### XGBoost Model
    xgb_model_path = f"s3://{default_bucket}/{prefix}/XGBoost_Model"
    xgb_image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.2-1",
    )
    
    xgb_estimator = Estimator(
        image_uri=xgb_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=xgb_model_path
    )
    
    xgb_estimator.set_hyperparameters(
        eval_metric="aucpr",
        objective="binary:logistic",
        eta=0.1,
        min_child_weight=2,
        max_depth=5,
        num_round=100,
        scale_pos_weight=10
    )
    ## ADD SCALING FACTOR HERE
    xgb_hyperparameter_ranges = {
        "eta": ContinuousParameter(0.01, 0.3),
        "min_child_weight": IntegerParameter(1, 5),
        "max_depth": IntegerParameter(3, 10),
        "num_round": IntegerParameter(50, 200),
        "scale_pos_weight": ContinuousParameter(1, 20)
    }
    
    xgb_tuner = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name="validation:aucpr",
        hyperparameter_ranges=xgb_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        random_seed=42,
        base_tuning_job_name="xgboost-tuning"
    )
    
    xgb_tuning_step = TuningStep(
        name="Tune-XGBoost",
        display_name="Tune XGBoost Model",
        tuner=xgb_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["xgboost"] = xgb_tuning_step
    
    ### Logistic Regression
    sklearn_image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
    )
    
    logit_model_path = f"s3://{default_bucket}/{prefix}/Logit_Model"
    logit_estimator = Estimator(
        image_uri=sklearn_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=logit_model_path,
        entry_point="s3://amazon-product-dataset-2024/scripts/logit_regression_train.py"
    )
    
    logit_hyperparameter_ranges = {
        "C": ContinuousParameter(0.01, 10.0),
        "penalty": CategoricalParameter(["l1", "l2", "elasticnet"]),
        "class_weight": CategoricalParameter(["balanced", "None"]),
        "solver": CategoricalParameter(["liblinear", "saga"])
    }
    
    logit_tuner = HyperparameterTuner(
        estimator=logit_estimator,
        objective_metric_name="validation_auc",
        hyperparameter_ranges=logit_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        base_tuning_job_name="logit-tuning"
    )
    
    logit_tuning_step = TuningStep(
        name="Tune-Logit",
        display_name="Tune Logistic Regression Model",
        tuner=logit_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["logit"] = logit_tuning_step
    
    ### Balanced random forest
    rf_model_path = f"s3://{default_bucket}/{prefix}/RF_Model"
    rf_estimator = Estimator(
        image_uri=sklearn_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=rf_model_path,
        entry_point="s3://amazon-product-dataset-2024/scripts/random_forest_train.py"
    )
    
    rf_hyperparameter_ranges = {
        "n_estimators": IntegerParameter(50, 300),
        "max_depth": IntegerParameter(3, 15),
        "min_samples_split": IntegerParameter(2, 20),
        "min_samples_leaf": IntegerParameter(1, 10),
        "class_weight": CategoricalParameter(["balanced", "balanced_subsample"])
    }
    
    rf_tuner = HyperparameterTuner(
        estimator=rf_estimator,
        objective_metric_name="validation_auc",
        hyperparameter_ranges=rf_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        base_tuning_job_name="rf-tuning"
    )
    
    rf_tuning_step = TuningStep(
        name="Tune-RF",
        display_name="Tune Balanced Random Forest Model",
        tuner=rf_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["rf"] = rf_tuning_step
    
    ### SVM
    svm_model_path = f"s3://{default_bucket}/{prefix}/SVM_Model"
    svm_estimator = Estimator(
        image_uri=sklearn_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=svm_model_path,
        entry_point="s3://amazon-product-dataset-2024/scripts/svm_train.py"
    )
    
    svm_hyperparameter_ranges = {
        "C": ContinuousParameter(0.01, 100.0),
        "kernel": CategoricalParameter(["linear", "rbf", "poly"]),
        "gamma": CategoricalParameter(["scale", "auto"]),
        "class_weight": CategoricalParameter(["balanced", "None"])
    }
    
    svm_tuner = HyperparameterTuner(
        estimator=svm_estimator,
        objective_metric_name="validation_auc",
        hyperparameter_ranges=svm_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        base_tuning_job_name="svm-tuning"
    )
    
    svm_tuning_step = TuningStep(
        name="Tune-SVM",
        display_name="Tune SVM Model",
        tuner=svm_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["svm"] = svm_tuning_step
    
    ### Isolation forest
    iso_model_path = f"s3://{default_bucket}/{prefix}/IsoForest_Model"
    iso_estimator = Estimator(
        image_uri=sklearn_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=iso_model_path,
        entry_point="s3://amazon-product-dataset-2024/scripts/isolation_forest_train.py"
    )
    
    iso_hyperparameter_ranges = {
        "n_estimators": IntegerParameter(50, 300),
        "max_samples": ContinuousParameter(0.1, 1.0),
        "contamination": ContinuousParameter(0.01, 0.3),
        "max_features": ContinuousParameter(0.1, 1.0)
    }
    
    iso_tuner = HyperparameterTuner(
        estimator=iso_estimator,
        objective_metric_name="validation_auc",
        hyperparameter_ranges=iso_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        base_tuning_job_name="isoforest-tuning"
    )
    
    iso_tuning_step = TuningStep(
        name="Tune-IsoForest",
        display_name="Tune Isolation Forest Model",
        tuner=iso_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["isoforest"] = iso_tuning_step
    
    ### One-class SVM
    ocsvm_model_path = f"s3://{default_bucket}/{prefix}/OCSVM_Model"
    ocsvm_estimator = Estimator(
        image_uri=sklearn_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=ocsvm_model_path,
        entry_point="s3://amazon-product-dataset-2024/scripts/one_class_svm_train.py"
    )
    
    ocsvm_hyperparameter_ranges = {
        "nu": ContinuousParameter(0.01, 0.5),
        "kernel": CategoricalParameter(["linear", "rbf", "poly"]),
        "gamma": CategoricalParameter(["scale", "auto"])
    }
    
    ocsvm_tuner = HyperparameterTuner(
        estimator=ocsvm_estimator,
        objective_metric_name="validation_auc",
        hyperparameter_ranges=ocsvm_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        base_tuning_job_name="ocsvm-tuning"
    )
    
    ocsvm_tuning_step = TuningStep(
        name="Tune-OCSVM",
        display_name="Tune One-Class SVM Model",
        tuner=ocsvm_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["ocsvm"] = ocsvm_tuning_step
    
    ### KNN
    knn_model_path = f"s3://{default_bucket}/{prefix}/KNN_Model"
    knn_estimator = Estimator(
        image_uri=sklearn_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=knn_model_path,
        entry_point="s3://amazon-product-dataset-2024/scripts/knn_train.py"
    )
    
    knn_hyperparameter_ranges = {
        "n_neighbors": IntegerParameter(3, 20),
        "weights": CategoricalParameter(["uniform", "distance"]),
        "p": IntegerParameter(1, 2),
        "metric": CategoricalParameter(["euclidean", "manhattan", "minkowski"])
    }
    
    knn_tuner = HyperparameterTuner(
        estimator=knn_estimator,
        objective_metric_name="validation_auc",
        hyperparameter_ranges=knn_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        base_tuning_job_name="knn-tuning"
    )
    
    knn_tuning_step = TuningStep(
        name="Tune-KNN",
        display_name="Tune KNN Model",
        tuner=knn_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["knn"] = knn_tuning_step
    
    ### Autoencoder
    tf_image_uri = sagemaker.image_uris.retrieve(
        framework="tensorflow",
        region=region,
        version="2.9.1",
        py_version="py39",
        instance_type=training_instance_type
    )
    
    autoencoder_model_path = f"s3://{default_bucket}/{prefix}/Autoencoder_Model"
    autoencoder_estimator = Estimator(
        image_uri=tf_image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        role=role,
        output_path=autoencoder_model_path,
        entry_point="s3://amazon-product-dataset-2024/scripts/autoencoder_train.py"
    )
    
    autoencoder_hyperparameter_ranges = {
        "encoding_dim": IntegerParameter(8, 64),
        "learning_rate": ContinuousParameter(0.0001, 0.01),
        "batch_size": IntegerParameter(32, 256),
        "epochs": IntegerParameter(10, 50),
        "dropout_rate": ContinuousParameter(0.1, 0.5)
    }
    
    autoencoder_tuner = HyperparameterTuner(
        estimator=autoencoder_estimator,
        objective_metric_name="validation_auc",
        hyperparameter_ranges=autoencoder_hyperparameter_ranges,
        max_jobs=max_jobs,
        max_parallel_jobs=max_parallel_jobs,
        strategy="Bayesian",
        early_stopping_type="Auto",
        base_tuning_job_name="autoencoder-tuning"
    )
    
    autoencoder_tuning_step = TuningStep(
        name="Tune-Autoencoder",
        display_name="Tune Autoencoder Model",
        tuner=autoencoder_tuner,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config
    )
    model_steps["autoencoder"] = autoencoder_tuning_step
    
    # Step 3: Model evaluation
    evaluation_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        base_job_name="model-evaluation",
        role=role,
        sagemaker_session=pipeline_session
    )
    
    # Iterate through model steps and create evaluation steps
    for model_name, tuning_step in model_steps.items():
        evaluation_step = ProcessingStep(
            name=f"Evaluate-{model_name.capitalize()}",
            display_name=f"Evaluate {model_name.capitalize()} Model",
            processor=evaluation_processor,
            inputs=[
                ProcessingInput(
                    source=tuning_step.properties.BestTrainingJob.ModelArtifacts.S3ModelArtifacts,
                    destination=f"/opt/ml/processing/model/{model_name}"
                ),
                ProcessingInput(
                    source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                    destination="/opt/ml/processing/test"
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="evaluation",
                    source=f"/opt/ml/processing/evaluation/{model_name}",
                    destination=Join(on="/", values=[f"s3://{default_bucket}", prefix, "evaluation", model_name])
                )
            ],
            code=f"s3://amazon-product-dataset-2024/scripts/evaluate_{model_name}.py",
            cache_config=cache_config
        )
        evaluation_steps[model_name] = evaluation_step
    
    # 4. Model Comparison Step
    comparison_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        base_job_name="model-comparison",
        role=role,
        sagemaker_session=pipeline_session
    )
    
    # Create inputs for the comparison step - include all evaluation outputs
    comparison_inputs = []
    for model_name, eval_step in evaluation_steps.items():
        comparison_inputs.append(
            ProcessingInput(
                source=eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                destination=f"/opt/ml/processing/evaluation/{model_name}"
            )
        )
    
    # comparison step
    model_comparison_step = ProcessingStep(
        name="Compare-Models",
        display_name="Compare All Models",
        processor=comparison_processor,
        inputs=comparison_inputs,
        outputs=[
            ProcessingOutput(
                output_name="comparison",
                source="/opt/ml/processing/comparison",
                destination=Join(on="/", values=[f"s3://{default_bucket}", prefix, "model_comparison"])
            )
        ],
        code="s3://amazon-product-dataset-2024/scripts/compare_models.py",
        cache_config=cache_config
    )
    
    # Step 5: Create pipeline
    pipeline_steps = [step_process]
    
    # Add model tuning steps
    for model_step in model_steps.values():
        pipeline_steps.append(model_step)
    
    # Add evaluation steps
    for eval_step in evaluation_steps.values():
        pipeline_steps.append(eval_step)
    
    # Add comparison step at the end
    pipeline_steps.append(model_comparison_step)

    return Pipeline(
        name=pipeline_name,
        parameters=[
            input_data,
            processing_instance_count,
            processing_instance_type,
            training_instance_type,
            max_jobs,
            max_parallel_jobs
        ],
        steps=pipeline_steps,
        sagemaker_session=pipeline_session)