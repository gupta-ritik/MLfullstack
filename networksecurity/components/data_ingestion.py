from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()
MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_collection_as_dataframe(self):
        """
        Fetch data from MongoDB → return DataFrame
        """
        try:
            logging.info("Connecting to MongoDB...")

            db_name = self.data_ingestion_config.database_name
            col_name = self.data_ingestion_config.collection_name

            client = pymongo.MongoClient(MONGO_DB_URL)
            collection = client[db_name][col_name]

            df = pd.DataFrame(list(collection.find()))
            logging.info(f"Fetched {df.shape[0]} rows from MongoDB database={db_name}, collection={col_name}")

            if df.empty:
                raise Exception(
                    f"MongoDB returned EMPTY dataset. "
                    f"Check database='{db_name}', collection='{col_name}'."
                )

            if "_id" in df.columns:
                df = df.drop(columns=["_id"])

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path

            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)

            dataframe.to_csv(feature_store_path, index=False, header=True)

            logging.info(f"Feature store saved at: {feature_store_path}")

            return dataframe

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logging.info(f"Splitting dataset with shape: {dataframe.shape}")

            if dataframe.empty:
                raise Exception("Cannot split EMPTY dataframe.")

            train_set, test_set = train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio,
                random_state=42
            )

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Train-test split completed successfully.")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion Pipeline...")

            df = self.export_collection_as_dataframe()
            df = self.export_data_into_feature_store(df)
            self.split_data_as_train_test(df)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

            logging.info("Data Ingestion Completed Successfully.")
            return artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
