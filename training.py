import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
import time


def lightgbm_train(train_X, train_y, test_X, test_y, ):
    clf = lgb.LGBMClassifier(
        boosting_type="gbdt",
        learning_rate=0.1,
        n_estimators=10,
        objective="binary",
        is_unbalance=True,
        seed=0,
        min_child_samples=1,
        min_child_weight=1)

    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    return pred_y == test_y

def RF_train(train_X, train_y, test_X, test_y, ):
    clf = RandomForestClassifier(n_estimators=10, max_depth=4, max_features=5, max_leaf_nodes=10)
    start = time.time()
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
    print("# rf training", time.time() - start)
    return pred_y != test_y


from sklearn.model_selection import StratifiedKFold


def cv_train(X, real_label, recorder, random_state=0, modifiy_label_method=None):
    if modifiy_label_method is None:
        modified_label = real_label
    else:
        modified_label, _ = modifiy_label_method(X, real_label)

    negative_count = len(real_label[real_label == 0])
    n_splits = min(negative_count, 5)
    cv = StratifiedKFold(n_splits=n_splits, random_state=random_state)
    fp = 0
    fn = 0
    for train_index, test_index in cv.split(X, modified_label):
        pred_result = lightgbm_train(X[train_index, 1:], modified_label[train_index], X[test_index, 1:],
                                     real_label[test_index])

        # aggregate prediction result compared to real label
        for idx, status in enumerate(pred_result):
            if status:
                continue
            fdate = X[test_index[idx]][0]
            if fdate not in recorder:
                recorder[fdate] = [0, 0]
            if real_label[test_index[idx]] == 0:
                recorder[fdate][0] += 1
                fp += 1
            else:
                recorder[fdate][1] += 1
                fn += 1

    print(fp, fn, len(recorder))
    return recorder

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, auc
import math
from collections import Counter
import numpy as np

def train_summary(clf, X, y):
    counter = Counter(y)
    print(counter)
    minority_class = min(counter, key=counter.get)
    pos_label = 1 - minority_class

    start = time.time()
    cv = StratifiedKFold(n_splits=5, random_state=10)
    result = []
    for i, split in zip(range(5), cv.split(X, y)):
        train_index, valid_index = split
        # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        train_X = X[train_index]
        test_X = X[valid_index]
        train_y = y[train_index]
        test_y = y[valid_index]
        clf.fit(train_X, train_y)
        pred_y = clf.predict_proba(test_X)[:, 1]
        # calculate auc
        # fpr_maj, tpr_maj, _ = roc_curve(test_y, pred_y, pos_label=pos_label)
        # fpr_min, tpr_min, _ = roc_curve(test_y, pred_y, pos_label=1 - pos_label)
        # auc_maj = auc(fpr_maj, tpr_maj)
        # auc_min = auc(fpr_min, tpr_min)
        auc_score = roc_auc_score(y[valid_index], pred_y)

        # calculate f-score
        pred_decision = np.where(pred_y > 0.5, [1], [0])
        fscore_maj = f1_score(test_y, pred_decision, average='binary', pos_label=pos_label)
        fscore_min = f1_score(test_y, pred_decision, average='binary', pos_label=1 - pos_label)

        # calculate g-mean
        tn, fp, fn, tp = confusion_matrix(test_y, pred_decision).ravel()
        acc = (tn + tp) / (tn + fp + fn + tp)
        tpr_maj = tp / (tp + fn)
        tnr = tn / (tn + fp)
        gmean = math.sqrt(tpr_maj * tnr)
        result.append([auc_score, fscore_maj, fscore_min, gmean])
        print([i, auc_score, fscore_maj, fscore_min, gmean, acc])
    print(np.array(result).mean(axis=0))

def cv(X, y):
    """
    only for the best combination of parameters
    :param X:
    :type X:
    :param y:
    :type y:
    :return:
    :rtype:
    """
    from sklearn.model_selection import GridSearchCV
    param_grid = {
        'learning_rate': [0.1],
        'n_estimators': [50, 100, 120],
        'max_depth': [5, 7, 9],
        'num_leaves': [10, 20, 30, 40]
    }
    grid3 = GridSearchCV(lgb.LGBMClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', verbose=100)
    grid3.fit(X, y)
    result = grid3.best_params_
    print(result)
    return result
