import numpy as np
from imbalance import *
from sampling import *
from sklearn.metrics import roc_auc_score

THRESHOLD = 0.13

def adaboost_train_1(X, y, train_index, valid_index, test_X, test_y, pos_label, c_iteration, max_features=-1):
    clf = lgb.LGBMClassifier(
        boosting_type="gbdt",
        learning_rate=0.1,
        n_estimators=40,
        max_depth=4,
        num_leaves=31,
        objective="binary",
        seed=0)

    start = time.time()
    clf.fit(X[train_index], y[train_index])

    pred_y = clf.predict_proba(test_X)[:, 1]
    pred_y = np.where(pred_y > THRESHOLD, [1], [0])

    # calculate auc
    auc_score = roc_auc_score(test_y, pred_y)

    # calculate f-score
    fscore_1 = metrics.f1_score(test_y, pred_y, average='binary', pos_label=1)
    fscore_0 = metrics.f1_score(test_y, pred_y, average='binary', pos_label=0)

    # calculate g-mean
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gmean = math.sqrt(tpr * tnr)
    print([time.time() - start, auc_score, fscore_0, fscore_1, gmean])
    return [auc_score, fscore_0, fscore_1, gmean]


def randomforest_train_1(X, y, train_index, valid_index, test_X, test_y, pos_label, c_iteration, max_features):
    clf = lgb.LGBMClassifier(
        boosting_type="rf",
        learning_rate=0.1,
        n_estimators=40,
        max_depth=4,
        num_leaves=31,
        objective="binary",
        seed=0,
        subsample=0.5,
        subsample_freq=1,
        feature_fraction=0.5,
    )
    start = time.time()
    clf.fit(X[train_index], y[train_index])
    pred_y = clf.predict_proba(test_X)[:, 1]
    pred_y = np.where(pred_y > THRESHOLD, [1], [0])
    # calculate auc
    auc_score = roc_auc_score(test_y, pred_y)

    # calculate f-score
    fscore_1 = metrics.f1_score(test_y, pred_y, average='binary', pos_label=1)
    fscore_0 = metrics.f1_score(test_y, pred_y, average='binary', pos_label=0)

    # calculate g-mean
    tn, fp, fn, tp = confusion_matrix(test_y, pred_y).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    gmean = math.sqrt(tpr * tnr)
    print([time.time() - start, auc_score, fscore_0, fscore_1, gmean])
    return [auc_score, fscore_0, fscore_1, gmean]

def comparison_test(repeat, method, full_X, full_y, pos_label, max_features, method_name=""):
    random_state = 0
    estimator_1 = adaboost_train_1
    estimator_2 = randomforest_train_1
    estimator = estimator_1
    ensemble_sampling = None

    if method == 1:
        ensemble_sampling = EasyEnsemble(random_state=random_state, n_subsets=4)
        estimator = estimator_1

    elif method == 0:
        ensemble_sampling = BalanceCascade(random_state=random_state, n_max_subset=4)
        estimator = estimator_1

    elif method == 2:
        ensemble_sampling = SMOTE(ratio='minority')
        estimator = estimator_1
    elif method == 3:
        ensemble_sampling = SMOTE(ratio='minority')
        estimator = estimator_2
    elif method == 4:
        ensemble_sampling = RandomOverSampler(ratio='minority', random_state=random_state)
        estimator = estimator_1
    elif method == 5:
        ensemble_sampling = RandomOverSampler(ratio='minority', random_state=random_state)
        estimator = estimator_2
    elif method == 6:
        ensemble_sampling = RandomUnderSampler(ratio='majority', random_state=random_state)
        estimator = estimator_1
    elif method == 7:
        ensemble_sampling = RandomUnderSampler(ratio='majority', random_state=random_state)
        estimator = estimator_2
    elif method == 8:
        ensemble_sampling = DummySampler()
        estimator = estimator_2
    elif method == 9:
        ensemble_sampling = DummySampler()
        estimator = estimator_1
    elif method >= 10:
        ratio = float(method_name.split("-")[1])
        ensemble_sampling = KSigmaSampler(target_ratio=ratio, random_state=random_state)
        if method_name.endswith("rf"):
            estimator = estimator_2
        else:
            estimator = estimator_1

    all_result = []
    if method == 0 or method == 1:
        repeat = 1
    for j in range(repeat):
        if method == 0 or method == 1:
            result = sampling_n_train_ensemble(full_X, full_y, ensemble_sampling, estimator=estimator,
                                               pos_label=pos_label, c_iteration=10)
        else:
            result = sampling_n_train(full_X, full_y, ensemble_sampling, estimator=estimator, pos_label=pos_label,
                                      c_iteration=40, max_features=max_features)
        all_result.append(result)

    return np.mean(all_result, axis=0)

import datetime

if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    data = ['pima', 'haberman', 'letter', 'credit card', 'santander']

    method = ['0BalanceCascade', '1EasyEnsemble', '2SMOTE+AdaBoost', '3SMOTE+RandomForest',
              '4RandomOverSampler+AdaBoost',
              '5RandomOverSampler+RandomForest', '6RandomUnderSampler+AdaBoost', '7RandomUnderSampler+RandomForest', '8Dummy-RF', '9Dummy-Boosting']

    santander_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.5, 2, 3, 4, 5, 10, 50, 100]
    for rate in santander_rates:
        method.append("3Sigma-{}-boost".format(rate))
    for rate in santander_rates:
        method.append("3Sigma-{}-rf".format(rate))

    testMethod = [i for i in range(len(method))]
    d_i = 4
    log_name = "imbalance-{}_1026.log".format(data[d_i])
    file_log(log_name, "start", str(datetime.datetime.now()))
    full_X, full_y, pos_label, max_features = generate_data(d_i)

    for m_i in testMethod:
        # if m_i != 9:
        #     continue
        # if m_i < 8:
        #     continue
        result = np.array(comparison_test(repeat=2, method=m_i, full_X=full_X, full_y=full_y, pos_label=pos_label, max_features=max_features, method_name=method[m_i]))
        file_log(log_name, data[d_i], method[m_i], result)
    file_log(log_name, "finish", str(datetime.datetime.now()))