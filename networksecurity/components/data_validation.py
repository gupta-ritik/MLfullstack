from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import os, sys


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            # Load schema.yaml
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("Schema.yaml loaded successfully.")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            logging.info(f"Reading data from: {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            schema_columns = self._schema_config["columns"]

            # Extract schema column names
            required_columns = [list(col.keys())[0] for col in schema_columns]

            logging.info(f"Required schema columns: {required_columns}")
            logging.info(f"DataFrame columns: {list(dataframe.columns)}")

            # Compare length
            if len(required_columns) != len(dataframe.columns):
                logging.error("Column count mismatch.")
                return False

            # Compare names
            if set(required_columns) != set(dataframe.columns):
                logging.error("Column names do not match schema.")
                return False

            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05):
        try:
            drift_status = True
            drift_report = {}

            logging.info("Starting dataset drift detection...")

            for column in base_df.columns:
                base_data = base_df[column]
                current_data = current_df[column]

                result = ks_2samp(base_data, current_data)
                p_value = float(result.pvalue)

                drift_found = p_value < threshold

                if drift_found:
                    drift_status = False   # drift detected

                drift_report[column] = {
                    "p_value": p_value,
                    "drift_found": drift_found
                }

            # Write drift report to YAML
            drift_report_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_path), exist_ok=True)
            write_yaml_file(drift_report_path, drift_report)

            logging.info(f"Drift report saved at: {drift_report_path}")

            return drift_status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating Data Validation Process...")

            # Load train & test datasets
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Schema validation
            if not self.validate_number_of_columns(train_df):
                raise Exception("Training dataset schema mismatch.")

            if not self.validate_number_of_columns(test_df):
                raise Exception("Testing dataset schema mismatch.")

            # Drift detection
            validation_status = self.detect_dataset_drift(train_df, test_df)

            # Save validated copies
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            logging.info("Validated train & test datasets saved successfully.")

            # Create artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info("Data Validation Completed Successfully.")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
