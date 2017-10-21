import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


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
    clf.fit(train_X, train_y)
    pred_y = clf.predict(test_X)
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
