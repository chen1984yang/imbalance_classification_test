import pandas as pd
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class KSigmaOutlierDetector(object):
    """
    Actually an abnormal detection among majority classes
    generally designed for binary classification
    """
    _k = 0
    _ofc = 1

    def __init__(self, k=3, outlying_feature_count=1):
        self._k = k
        self._ofc = outlying_feature_count

    def fit_predict(self, X, y):
        counter = Counter(y)
        minority_class = min(counter, key=counter.get)

        outlying_count = np.zeros(shape=(X.shape[0]))
        for i in range(X.shape[1]):
            mean = np.mean(X[:, i])
            std = np.std(X[:, i])
            for j in range(X.shape[0]):
                if y[j] == minority_class:
                    continue
                if abs(X[j, i] - mean) < std * self._k:
                    outlying_count[j] += 1
        return np.where(outlying_count > self._ofc, [-1], [1])


class SimpleOutlierDetector(object):
    """
    Actually an abnormal detection among majority classes
    mark
    generally designed for binary classification
    """
    _k = 0
    _ofc = 1

    def __init__(self, k=3, outlying_feature_count=1):
        self._k = k
        self._ofc = outlying_feature_count

    def fit_predict(self, X, y):
        counter = Counter(y)
        _, minority_class = min(counter, key=counter.get)

        if type(X) == pd.DataFrame:
            X = X.values
        positives = X[y == 1]
        negatives = X[y == 0]
        groundtruth = np.ones_like(y)
        updated_count = np.zeros_like(y)
        for k in range(1, X.shape[1]):
            p_mean = np.mean(positives[:, k])
            n_mean = np.mean(negatives[:, k])
            p_sigma = max(np.std(positives[:, k]), 1e-9)
            n_sigma = max(np.std(negatives[:, k]), 1e-9)
            for i, j in enumerate(y):
                if j == minority_class:  # if is positive
                    continue
                if abs(X[i, k] - p_mean) * n_sigma < abs(X[i, k] - n_mean) * p_sigma:
                    updated_count[i] += 1
        for i, j in enumerate(updated_count):
            if j != 0:
                groundtruth[i] = -1
        return groundtruth


class DummyOutlierDetector(object):
    def __init__(self):
        pass

    def fit_predict(self, X):
        return np.ones(shape=(X.shape[0]))


def generate_detectors(n_samples, n_features, estimated_outlier_fraction=0.05):
    classifiers = {
        # "One-Class-SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
        #                                  kernel="rbf", gamma=0.1),
        "Robust-Covariance": EllipticEnvelope(contamination=estimated_outlier_fraction),
        "Isolation-Forest": IsolationForest(max_samples=n_samples,
                                            contamination=estimated_outlier_fraction,
                                            random_state=42),
        "Local-Outlier-Factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=estimated_outlier_fraction),
        "Dummy": DummyOutlierDetector()
    }
    # add a bunch of ksigma-outlier-detector
    for j in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6]:
        n_outlying_feature = int(j * n_features)
        classifiers["3Sigma-{}".format(n_outlying_feature)] = KSigmaOutlierDetector(
            outlying_feature_count=n_outlying_feature)

    return classifiers
