import sys

from networksecurity.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation


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

        logging.info("========== PIPELINE COMPLETED SUCCESSFULLY ==========")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
