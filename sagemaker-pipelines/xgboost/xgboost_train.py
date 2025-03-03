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
import logging

session = sagemaker.session.Session()
region = session.boto_region_name
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession()
default_bucket = "amazon-product-dataset-2024"
prefix = "model-comparison"

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def get_pipeline(pipeline_name):
    logger.info("Initializing pipeline parameters")
    input_data = ParameterString(name="Input-Data", 
                                 default_value=f"s3://{default_bucket}/transformed/transformed-data.parquet")
       
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
    cache_config = CacheConfig(enable_caching=True, expire_after="30d")
    
    logger.info("Creating SKLearnProcessor")
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="xgboost-processing",
        role=role,
        sagemaker_session=session
    )

    logger.info("Creating ProcessingStep for data preparation")
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
        cache_config=cache_config)
        
    logger.info("Setting up training parameters")
    final_model_path = f"s3://{default_bucket}/{prefix}"

    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.2-1",
        )
    logger.info("Retrieved XGBoost image URI")
    
    xgb_estimator = Estimator(
        image_uri=image_uri,
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        role=role,
        output_path=final_model_path)
    
    xgb_estimator.set_hyperparameters(
        eval_metric="aucpr",
        objective="binary:logistic",
        eta=0.01,
        min_child_weight=5,
        max_depth=5,
        num_round=250,
        scale_pos_weight=10)
    
    logger.info("Set XGBoost estimator hyperparameters")
    
    hyperparameter_ranges = {
        "eta": ContinuousParameter(0.01, 0.2), 
        "min_child_weight": IntegerParameter(1, 3),
        "max_depth": IntegerParameter(3, 10),
        "num_round": IntegerParameter(50, 55), 
        "scale_pos_weight": ContinuousParameter(20, 22)
    }
    
    logger.info("Initializing HyperparameterTuner")

    xgb_tuner = HyperparameterTuner(
        estimator=xgb_estimator,
        objective_metric_name="validation:aucpr",
        hyperparameter_ranges=hyperparameter_ranges,
        max_jobs=5,
        max_parallel_jobs=2,
        strategy="Bayesian",
        early_stopping_type="Auto",
        random_seed=42,
        base_tuning_job_name="xgboost-tuning"
    )
    
    logger.info("Initializing TuningStep")

    step_tuning = TuningStep(
        name="Train-And-Tune-Model",
        display_name="Train And Tune Model",
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
    
   
    logger.info("Tuning complete, moving on to evaluation")

    logger.info("Creating SKLearnProcessor for evaluation")
    evaluation_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        base_job_name="xgboost-evaluation",
        role=role,
        sagemaker_session=session)
    
    logger.info("Creating ProcessingStep for model evaluation")
    step_evaluate = ProcessingStep(
        name="Evaluate-Model",
        display_name="Evaluate Model",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=step_tuning.get_top_model_s3_uri(top_k=1,
                                                        s3_bucket=default_bucket,
                                                        prefix=prefix),
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test")],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation",
                destination=Join(on="/", values=[f"s3://{default_bucket}", prefix, "evaluation"]))],
        code="s3://amazon-product-dataset-2024/scripts/evaluation.py")

    # logger.info("Returning the pipeline object")
    return Pipeline(
        name=pipeline_name,
        parameters=[
            input_data, processing_instance_count, processing_instance_type
        ],
        steps=[step_process, step_tuning, step_evaluate],
        sagemaker_session=session)
