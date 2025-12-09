import sys
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.components.data_ingestion import DataIngestion


if __name__ == "__main__":
    try:
        logging.info("Starting Training Pipeline Configuration")

        # Load global pipeline config
        training_pipeline_config = TrainingPipelineConfig()

        # Load Data Ingestion related config
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        # Create DataIngestion object
        data_ingestion = DataIngestion(data_ingestion_config)

        logging.info("Initiating Data Ingestion Process")

        # Start the data ingestion pipeline
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        print("\n===== DATA INGESTION ARTIFACT =====")
        print(data_ingestion_artifact)

        logging.info("Data Ingestion Completed Successfully")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
