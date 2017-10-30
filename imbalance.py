from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import *
from imblearn.ensemble import EasyEnsemble, BalanceCascade
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
import numpy as np
import time
import math
import lightgbm as lgb
import dataset
from collections import Counter

from logger import *

def adaboost_train(X, y, train_index, valid_index, test_X, test_y, pos_label, c_iteration, max_features=-1):
    clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=c_iteration)
    start = time.time()
    clf.fit(X[train_index], y[train_index])

    pred_y = clf.predict(test_X)
    # calculate auc
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)

    # calculate f-score
    fscore = metrics.f1_score(test_y, pred_y, average='binary')

    # calculate g-mean
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gmean = math.sqrt(tpr * tnr)
    print([time.time() - start, auc, fscore, gmean])
    return [auc, fscore, gmean]


def randomforest_train(X, y, train_index, valid_index, test_X, test_y, pos_label, c_iteration, max_features):
    clf = RandomForestClassifier(n_estimators=c_iteration, max_features=max_features, n_jobs=4)
    start = time.time()
    clf.fit(X[train_index], y[train_index])
    pred_y = clf.predict(test_X)
    # calculate auc
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)

    # calculate f-score
    fscore = metrics.f1_score(test_y, pred_y, average='binary')

    # calculate g-mean
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gmean = math.sqrt(tpr * tnr)
    print([time.time() - start, auc, fscore, gmean])
    return [auc, fscore, gmean]


def cv_train(X, y, test_X, test_y, method, pos_label, c_iteration, max_features, random_state=0):
    cv = StratifiedKFold(n_splits=10, random_state=random_state)
    conf_mat = []
    for train_index, valid_index in cv.split(X, y):
        start = time.time()
        cf = method(X, y, train_index, valid_index, test_X, test_y, pos_label, c_iteration, max_features)
        # file_log("imbalance.log", "training time", time.time() - start)
        conf_mat.append(cf)
    return np.mean(conf_mat, axis=0)

from sklearn.model_selection import train_test_split

def sampling_n_train(full_X, full_y, sampling_method, estimator, pos_label, c_iteration, max_features, repeat=4):
    start = time.time()
    time_used = time.time() - start
    print(sampling_method.__class__.__name__, "sampling time", time_used)
    # file_log("imbalance.log", "sampling ended", sampling_method.__class__.__name__, pos_count, neg_count, time_used)

    conf_mat = []
    poss = []
    negs = []
    for j in range(repeat):
        train_X, test_X, train_y, test_y = train_test_split(full_X, full_y, test_size=1 / repeat, stratify=full_y)
        # neg_count = len(train_y[train_y == 0])
        # pos_count = len(train_y) - neg_count
        # file_log("imbalance.log", "sampling started", sampling_method.__class__.__name__, pos_count, neg_count)

        X, y = sampling_method.fit_sample(train_X, train_y)
        sampled_neg_count = len(y[y == 0])
        sampled_pos_count = len(y) - sampled_neg_count
        poss.append(sampled_pos_count)
        negs.append(sampled_neg_count)

        conf_mat.append(cv_train(X, y, test_X, test_y, estimator, pos_label, c_iteration, max_features, random_state=j))
    totalTime = time.time() - start
    n_trials = len(conf_mat)
    avg_sampled_pos_count = np.mean(poss)
    avg_sampled_neg_count = np.mean(negs)
    result1 = [time_used / n_trials, totalTime / n_trials, avg_sampled_pos_count, avg_sampled_neg_count]
    result2 = np.mean(conf_mat, axis=0)
    result = []
    result.extend(result1)
    result.extend(result2)
    return result


def sampling_n_train_ensemble(full_X, full_y, sampling_method, estimator, pos_label, c_iteration, repeat=4, random_state=0):
    start = time.time()

    used = time.time() - start
    results = []
    negs = []
    poss = []
    for i in range(repeat):
        train_X, test_X, train_y, test_y = train_test_split(full_X, full_y, test_size=1/repeat)
        X_list, y_list = sampling_method.fit_sample(train_X, train_y)
        for X, y in zip(X_list, y_list):
            neg_count = len(y[y == 0])
            pos_count = len(y[y == 1])
            negs.append(neg_count)
            poss.append(pos_count)
            results.append(cv_train(X, y, test_X, test_y, estimator, pos_label, c_iteration, max_features=-1, random_state=random_state))
    totalTime = time.time() - start
    n_trials = len(results)
    print("trial result", np.array(results).mean(axis=0), len(full_y))
    result1 = [used / n_trials, totalTime / n_trials]
    result2 = np.mean(results, axis=0) * repeat
    result = []
    result.extend(result1)
    result.extend(result2)
    return result
    # return used, np.mean(poss), np.mean(negs), np.mean(conf_mat, axis=0)

import os
from os.path import join
location = os.path.dirname(os.path.abspath(__file__))

def generate_data(data_index):
    data = None
    full_X, full_y = None, None
    pos_label = 1
    delimiter = ','

    if data_index == 0:
        data = np.loadtxt(join(location, 'data/pima.data'), delimiter=delimiter)
        full_X = data[:, 0:8]
        full_y = data[:, 8]
        pos_label = 1
        full_y = full_y.astype(int)

    elif data_index == 1:
        data = np.loadtxt(join(location, 'data/haberman.data'), delimiter=delimiter)
        full_X = data[:, 0:3]
        full_y = data[:, 3]
        pos_label = 2
        full_y = full_y.astype(int)
    elif data_index == 2:
        full_X = np.loadtxt(join(location, 'data/letter-recognition.data'), delimiter=delimiter,
                            usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
        labels = np.loadtxt(join(location, 'data/letter-recognition.data'), delimiter=delimiter, usecols=[0], dtype=np.str)
        full_y = []
        for x in labels:
            v = 1 if x == 'A' else 0
            full_y.append(v)
        pos_label = 1
    elif data_index == 3:
        data = np.loadtxt(join(location, 'data', 'FraudDetection', 'creditcard.csv'), delimiter=delimiter, skiprows=1, dtype=bytes)
        # sample = []
        # for j in range(100):
        #    sample.append(data[j])

        data = np.array(data)
        full_X = data[:, 0:30].astype(float)
        labels = data[:, 30]
        full_y = []
        for x in labels:
            v = 0 if x == b'"0"' else 1
            full_y.append(v)
        pos_label = 1

    elif data_index == 4:
        full_X, full_y = dataset.load_santander()
        pos_label = 0

    max_features = full_X.shape[1]
    file_log("imbalance.log", 'Features: ', max_features)
    return full_X, np.array(full_y), pos_label, max_features

