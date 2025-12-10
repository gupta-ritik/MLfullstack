import sys

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer


if __name__ == "__main__":
    try:
        logging.info("========== PIPELINE STARTED ==========")

        # Load global pipeline configuration
        training_pipeline_config = TrainingPipelineConfig()

        # -------------------------------
        # Step 1: Data Ingestion
        # -------------------------------
        logging.info("Initializing Data Ingestion...")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion Completed Successfully.")
        print("\nData Ingestion Artifact:\n", data_ingestion_artifact)

        # -------------------------------
        # Step 2: Data Validation
        # -------------------------------
        logging.info("Initializing Data Validation...")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)

        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation Completed Successfully.")
        print("\nData Validation Artifact:\n", data_validation_artifact)

        # -------------------------------
        # Step 3: Data Transformation
        # -------------------------------
        logging.info("Initializing Data Transformation...")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)

        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data Transformation Completed Successfully.")
        print("\nData Transformation Artifact:\n", data_transformation_artifact)

        # -------------------------------
        # Step 4: Model Training
        # -------------------------------
        logging.info("Initializing Model Training...")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)

        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )

        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model Training Completed Successfully.")
        print("\nModel Trainer Artifact:\n", model_trainer_artifact)

        logging.info("========== PIPELINE COMPLETED SUCCESSFULLY ==========")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
