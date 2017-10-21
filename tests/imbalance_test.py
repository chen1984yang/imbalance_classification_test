import numpy as np
from imbalance import *


def comparison_test(repeat, method, full_X, full_y, pos_label, max_features):
    random_state = 0
    estimator = adaboost_train
    ensemble_sampling = None

    if method == 1:
        ensemble_sampling = EasyEnsemble(random_state=random_state, n_subsets=4)
        estimator = adaboost_train

    elif method == 0:
        ensemble_sampling = BalanceCascade(random_state=random_state, n_max_subset=4)
        estimator = adaboost_train

    elif method == 2:
        ensemble_sampling = SMOTE(ratio='minority')
        estimator = adaboost_train
    elif method == 3:
        ensemble_sampling = SMOTE(ratio='minority')
        estimator = randomforest_train
    elif method == 4:
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


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    data = ['pima', 'haberman', 'letter', 'credit card']

    method = ['0BalanceCascade', '1EasyEnsemble', '2SMOTE+AdaBoost', '3SMOTE+RandomForest',
              '4RandomOverSampler+AdaBoost',
              '5RandomOverSampler+RandomForest', '6RandomUnderSampler+AdaBoost', '7RandomUnderSampler+RandomForest']
    testMethod = [i for i in range(len(method))]
    d_i = 1
    full_X, full_y, pos_label, max_features = generate_data(d_i)

    for m_i in testMethod:
        result = np.array(comparison_test(repeat=5, method=m_i, full_X=full_X, full_y=full_y, pos_label=pos_label,
                                          max_features=max_features))
        file_log("imbalance-{}.log".format(data[d_i]), data[d_i], method[m_i], result)
