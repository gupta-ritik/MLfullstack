import yaml
import os
import sys
import numpy as np
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging


# ============================
# YAML READ / WRITE
# ============================

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "r") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ============================
# NUMPY SAVE / LOAD
# ============================

def save_numpy_array_data(file_path: str, array: np.array):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ============================
# OBJECT SAVE / LOAD
# ============================

def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Saving object...")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Object saved successfully.")

    except Exception as e:
        raise NetworkSecurityException(e, sys)


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File not found: {file_path}")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys)


# ============================
# MODEL EVALUATION
# (Classification)
# ============================

def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid):
    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import f1_score

        report = {}

        for model_name, model in models.items():

            params = param_grid.get(model_name, {})

            gs = GridSearchCV(model, params, cv=3, verbose=0)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)

            score = f1_score(y_test, y_pred, zero_division=0)

            report[model_name] = {
                "best_model": best_model,
                "f1_score": score
            }

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)
