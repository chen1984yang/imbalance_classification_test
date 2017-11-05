import pandas as pd
import numpy as np
from collections import Counter
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
class KSigmaOutlierDetector(object):
    """
    Actually an abnormal detection among majority classes
    generally designed for binary classification
    """
    _k = 0
    _ofc = 1


    def __init__(self, k=3, outlying_feature_count=1, contamination=0.05):
        self._k = k
        self._ofc = outlying_feature_count

    def fit_predict(self, X):
        outlying_count = np.zeros(shape=(X.shape[0]))
        for i in range(X.shape[1]):
            mean = np.mean(X[:, i])
            std = np.std(X[:, i])
            for j in range(X.shape[0]):
                if abs(X[j, i] - mean) >= std * self._k:
                    outlying_count[j] += 1
        return np.where(outlying_count > self._ofc, [-1], [1])

    def set_params(self, **params):
        pass

#
# class SimpleOutlierDetector(object):
#     """
#     Actually an abnormal detection among majority classes
#     mark
#     generally designed for binary classification
#     """
#     _k = 0
#     _ofc = 1
#
#     def __init__(self, k=3, outlying_feature_count=1):
#         self._k = k
#         self._ofc = outlying_feature_count
#
#     def fit_predict(self, X, y):
#         counter = Counter(y)
#         _, minority_class = min(counter, key=counter.get)
#
#         if type(X) == pd.DataFrame:
#             X = X.values
#         positives = X[y == 1]
#         negatives = X[y == 0]
#         groundtruth = np.ones_like(y)
#         updated_count = np.zeros_like(y)
#         for k in range(1, X.shape[1]):
#             p_mean = np.mean(positives[:, k])
#             n_mean = np.mean(negatives[:, k])
#             p_sigma = max(np.std(positives[:, k]), 1e-9)
#             n_sigma = max(np.std(negatives[:, k]), 1e-9)
#             for i, j in enumerate(y):
#                 if j == minority_class:  # if is positive
#                     continue
#                 if abs(X[i, k] - p_mean) * n_sigma < abs(X[i, k] - n_mean) * p_sigma:
#                     updated_count[i] += 1
#         for i, j in enumerate(updated_count):
#             if j != 0:
#                 groundtruth[i] = -1
#         return groundtruth


class DummyOutlierDetector(object):
    def __init__(self):
        pass

    def fit_predict(self, X):
        return np.ones(shape=(X.shape[0]))

    def set_params(self, **params):
        pass

def generate_ksigma_detectors(n_features):
    classifiers = {}
    # add a bunch of ksigma-outlier-detector
    for j in range(2):
    # for j in [0.1]:
        ratio_threshold = (j + 1) * 0.5
        n_outlying_feature = int(ratio_threshold * n_features)

        classifiers["3Sigma-{}".format(n_outlying_feature)] = KSigmaOutlierDetector(
            outlying_feature_count=n_outlying_feature)
    return classifiers


def generate_detectors(n_samples, n_features, estimated_outlier_fraction=0.05, random_state=0):
    classifiers = {
        # "One-Class-SVM": svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05,
        #                                  kernel="rbf", gamma=0.1),
        "Robust-Covariance": EllipticEnvelope(contamination=estimated_outlier_fraction, random_state=random_state),
        "Isolation-Forest": IsolationForest(contamination=estimated_outlier_fraction, random_state=random_state),
        "Local-Outlier-Factor": LocalOutlierFactor(
            n_neighbors=35,
            contamination=estimated_outlier_fraction),
        # "DBScan": DBSCAN(),

        "Dummy": DummyOutlierDetector()
    }
    ksigma_detectors = generate_ksigma_detectors(n_features)
    classifiers.update(ksigma_detectors)

    return classifiers


def omni_detector_detect(detector, X) -> np.ndarray:
    detector_class = detector.__class__
    if detector_class == KSigmaOutlierDetector:
        y_pred = detector.fit_predict(X)
    else:
        if detector_class == LocalOutlierFactor or detector_class == DummyOutlierDetector:
            y_pred = detector.fit_predict(X)
        else:
            detector.fit(X)
            y_pred = detector.predict(X)
    return y_pred.astype(np.int)

from dataset import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from active_modify_sample_label import *
from tool import *

def update_by_outlier_prediction(original_y, outlier_prediction):
    updated_y = []
    for i, l in enumerate(outlier_prediction):
        if l == -1:
            updated_y.append(1 - original_y[i])
        else:
            updated_y.append(original_y[i])
    return updated_y

def detection_n_modify(detector_name, detector_method, X, y, groundtruth):
    counter = Counter(y)
    minority_class = min(counter, key=counter.get)
    minority_X = X[y==minority_class]
    if detector_name == "Local-Outlier-Factor" \
            or detector_name == "Dummy"\
            or detector_name.startswith("3Sigma-"):
        outlier_prediction = detector_method.fit_predict(minority_X)
    else:
        detector_method.fit(minority_X)
        outlier_prediction = detector_method.predict(minority_X)

    conf_mat = confusion_matrix(groundtruth, outlier_prediction)
    n_errors = conf_mat[0][1] + conf_mat[1][0]
    print(detector_name, list2str(np.ravel(conf_mat)), n_errors)

    minority_updated_y = update_by_outlier_prediction(y[y==minority_class], outlier_prediction)
    updated_y = np.array(y)
    updated_y[y==minority_class] = minority_updated_y
    return updated_y


if __name__ == '__main__':
    d = LocalOutlierFactor(
            n_neighbors=35,
            contamination=0.05)
    d.set_params(contamination=0.2)
    noise_ratio = 0.2
    seed = 40

    X, y = load_santander()
    n_samples, n_features = X.shape
    detectors = generate_detectors(n_samples, n_features, noise_ratio, random_state=seed)
    param = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 100,
        'num_leaves': 30
    }
    model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        learning_rate=param["learning_rate"],
        n_estimators=param["n_estimators"],
        max_depth=param["max_depth"],
        num_leaves=param["num_leaves"],
        objective="binary"
    )
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=seed)
    noised_train_y, groundtruth = add_noise_to_majority(train_y, noise_ratio, random_state=seed, verbose=True)
    for detector_name in detectors:
        if not detector_name.startswith("3Sigma-"):
            continue
        detection_n_modify(detector_name, detectors[detector_name], train_X, train_y, groundtruth)
