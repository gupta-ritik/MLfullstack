import os
import sys
import mlflow
import mlflow.sklearn

from networksecurity.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.utils.main_utils.utils import evaluate_models, load_object, load_numpy_array_data, save_object
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifact = data_transformation_artifact

    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=250),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Random Forest": {"n_estimators": [32, 64]},
            "Decision Tree": {"criterion": ["gini", "entropy"]},
            "Gradient Boosting": {"learning_rate": [0.1, 0.05], "n_estimators": [32, 64]},
            "Logistic Regression": {},
            "AdaBoost": {"learning_rate": [0.1, 0.01], "n_estimators": [32, 64]},
        }

        # GET FIXED MODEL REPORT
        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param_grid=params
        )

        # FIND BEST MODEL
        best_model_name = max(model_report, key=lambda x: model_report[x]["f1_score"])
        best_model = model_report[best_model_name]["best_model"]

        logging.info(f"Best model selected: {best_model_name}")
        save_object("final_model/best_model.pkl", best_model)
        # Compute train and test metrics
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metric = get_classification_score(y_train, y_train_pred)
        test_metric = get_classification_score(y_test, y_test_pred)

        # Load transformer
        preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)

        # Build final pipeline model
        final_model = NetworkModel(preprocessor, best_model)

        save_object(self.model_trainer_config.trained_model_file_path, final_model)

        # Return artifact
        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric
        )

    def initiate_model_trainer(self):

        train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
        test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        return self.train_model(X_train, y_train, X_test, y_test)
