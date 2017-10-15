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


def adaboost_train(X, y, train_index, valid_index, test_X, test_y, pos_label,c_iteration, max_features=-1):
    clf = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=c_iteration)
    clf.fit(X[train_index], y[train_index])

    pred_y = clf.predict(test_X)
    #calculate auc
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)

    #calculate f-score
    fscore = metrics.f1_score(test_y, pred_y, average='binary')

    #calculate g-mean
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    gmean = math.sqrt(tpr*tnr)

    return [auc, fscore, gmean]


def randomforest_train(X, y, train_index, valid_index, test_X, test_y, pos_label,c_iteration,max_features):
    clf = RandomForestClassifier(n_estimators=c_iteration, max_depth=4, max_features=max_features, max_leaf_nodes=10)
    clf.fit(X[train_index], y[train_index])

    pred_y = clf.predict(test_X)
    #calculate auc
    fpr, tpr, thresholds = metrics.roc_curve(test_y, pred_y, pos_label=pos_label)
    auc = metrics.auc(fpr, tpr)

    #calculate f-score
    fscore = metrics.f1_score(test_y, pred_y, average='binary')

    #calculate g-mean
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    gmean = math.sqrt(tpr*tnr)

    return [auc, fscore, gmean]


def cv_train(X, y, test_X, test_y, method, pos_label,c_iteration,max_features, random_state=0):
    cv = StratifiedKFold(n_splits=10, random_state=random_state)
    conf_mat = []
    for train_index, valid_index in cv.split(X, y):
        cf = method(X, y, train_index, valid_index, test_X, test_y, pos_label,c_iteration,max_features)
        conf_mat.append(cf)
    return np.mean(conf_mat, axis=0)


def sampling_n_train(full_X, full_y, sampling_method, estimator, pos_label, c_iteration, max_features, repeat=10):
    counts = []
    start = time.time()
    X, y = sampling_method.fit_sample(full_X, full_y)
    neg_count = len(y[y == 0])
    pos_count = len(y[y == 1])
    time_used = time.time() - start
    counts.append(len(y))
    conf_mat = []
    for j in range(repeat):
        conf_mat.append(cv_train(X, y, full_X, full_y, estimator, pos_label, c_iteration, max_features, random_state=j))
    totalTime = time.time() - start
    result1 = [time_used, totalTime,pos_count, neg_count]
    result2 = np.mean(conf_mat, axis=0)
    result = []
    result.extend(result1)
    result.extend(result2)
    return result


def sampling_n_train_ensemble(full_X, full_y, sampling_method, estimator, pos_label, c_iteration,random_state=0):
    start = time.time()

    X_list, y_list = sampling_method.fit_sample(full_X, full_y)

    used = time.time() - start
    conf_mat = []
    negs = []
    poss = []
    for X, y in zip(X_list, y_list):
        neg_count = len(y[y == 0])
        pos_count = len(y[y == 1])
        negs.append(neg_count)
        poss.append(pos_count)
        conf_mat.append(cv_train(X, y, full_X, full_y, estimator, pos_label, c_iteration, max_features = -1,random_state=random_state))
    totalTime = time.time() - start
    result1 = [used, totalTime,np.mean(poss), np.mean(negs)]
    result2 = np.mean(conf_mat, axis=0)
    result = []
    result.extend(result1)
    result.extend(result2)
    return result
    #return used, np.mean(poss), np.mean(negs), np.mean(conf_mat, axis=0)

def generate_data(data_index):
    data = None
    full_X, full_y = None, None
    pos_label = 1
    delimiter = ','

    if data_index==0:
       data = np.loadtxt('data/pima.data', delimiter=delimiter)
       full_X = data[:, 0:8]
       full_y = data[:, 8]
       pos_label = 1
       full_y = full_y.astype(int)

    elif data_index ==1:
        data = np.loadtxt('data/haberman.data', delimiter=delimiter)
        full_X = data[:, 0:3]
        full_y = data[:, 3]
        pos_label = 2
        full_y = full_y.astype(int)
    elif data_index == 2:
        full_X = np.loadtxt('data/letter-recognition.data', delimiter=delimiter, usecols=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        labels = np.loadtxt('data/letter-recognition.data', delimiter=delimiter, usecols=[0],dtype=np.str)
        full_y = []
        for x in labels:
            v = 1 if x == 'A' else 0
            full_y.append(v)
        pos_label = 1
    elif data_index==3:
        data = np.loadtxt('data/creditcard.csv', delimiter=delimiter,skiprows=1,dtype=np.str)
        #sample = []
        #for j in range(100):
        #    sample.append(data[j])

        data = np.array(data)
        full_X = data[:, 0:30].astype(float)
        labels = data[:, 30]
        full_y = []
        for x in labels:
            v = 0 if x == '"0"' else 1
            full_y.append(v)
        pos_label = 1

    max_features = full_X.shape[1]
    print('Features: ', max_features)
    return full_X,full_y,pos_label,max_features

def comparison_test(repeat, method, full_X, full_y, pos_label,max_features):

    random_state = 0
    estimator = adaboost_train
    ensemble_sampling = None


    if method == 1:
        ensemble_sampling = EasyEnsemble(random_state=random_state, n_subsets=4)
        estimator = adaboost_train

    elif method == 0:
        ensemble_sampling = BalanceCascade(random_state=random_state, n_max_subset=4)
        estimator = adaboost_train

    elif method ==2:
        ensemble_sampling = SMOTE(ratio='minority')
        estimator = adaboost_train
    elif method ==3:
        ensemble_sampling = SMOTE(ratio='minority')
        estimator = randomforest_train
    elif method ==4:
        ensemble_sampling = RandomOverSampler(ratio='minority', random_state=random_state)
        estimator = adaboost_train
    elif method == 5:
        ensemble_sampling = RandomOverSampler(ratio='minority', random_state=random_state)
        estimator = randomforest_train
    elif method == 6:
        ensemble_sampling = RandomUnderSampler(ratio='majority', random_state=random_state)
        estimator = adaboost_train
    elif method == 7:
        ensemble_sampling = RandomUnderSampler(ratio='majority', random_state=random_state)
        estimator = randomforest_train

    all_result = []
    for j in range(repeat):
        if method==0 or method ==1:
            result = sampling_n_train_ensemble(full_X, full_y, ensemble_sampling,estimator=estimator, pos_label = pos_label, c_iteration=10)
        else:
            result = sampling_n_train(full_X, full_y, ensemble_sampling,estimator=estimator, pos_label =pos_label,c_iteration=40, max_features=max_features)
        all_result.append(result)

    return np.mean(all_result, axis=0)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    data = ['pima','haberman','letter','credit card']

    method = ['0BalanceCascade', '1EasyEnsemble','2SMOTE+AdaBoost','3SMOTE+RandomForest','4RandomOverSampler+AdaBoost',
              '5RandomOverSampler+RandomForest','6RandomUnderSampler+AdaBoost','7RandomUnderSampler+RandomForest']
    testMethod = [0]
    #testMethod = [0]
    d_i = 3
    full_X, full_y, pos_label,max_features = generate_data(d_i)


    for m_i in testMethod:
        result = np.array(comparison_test(repeat=5, method=m_i, full_X=full_X, full_y=full_y,pos_label=pos_label,max_features=max_features))
        print (data[d_i] ,method[m_i], result)