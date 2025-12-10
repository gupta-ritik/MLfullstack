from networksecurity.entity.artifact_entity import ClassificationMetricArtifact
from networksecurity.exception.exception import NetworkSecurityException
from sklearn.metrics import f1_score, recall_score, precision_score
import sys


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        return ClassificationMetricArtifact(
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            precision_score=precision_score(y_true, y_pred, zero_division=0),
            recall_score=recall_score(y_true, y_pred, zero_division=0),
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)
