import random

from dataset import *
from logger import *
from training import *


def modify_label_dummy(X, y):
    name = "dummy"
    return y, np.ones_like(y)


from collections import Counter


def add_noise_binary(y, noise_ratio, random_state=0):
    y = np.array(y)
    groundtruth = np.ones_like(y)
    random.seed(random_state * 2)
    counter = Counter(y)
    m = [-1, -1]
    for k in counter:
        v = counter[k]
        if v > m[1]:
            m = [k, v]

    class_labels = list(counter.keys())
    class_count = len(class_labels)

    majority_class = m[0]
    outlier_count = 0
    for i, label in enumerate(y):
        if label != majority_class:
            continue
        if random.random() < noise_ratio:
            y[i] = class_labels[int(random.random() * class_count)]
            groundtruth[i] = - 1
            outlier_count += 1
    print(outlier_count, "outlier added")
    return y, groundtruth


from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix


def update_by_outlier_prediction(original_y, outlier_prediction):
    updated_y = []
    for i, l in enumerate(outlier_prediction):
        if l == -1:
            updated_y.append(1 - original_y[i])
        else:
            updated_y.append(original_y[i])
    return updated_y


def active_modify_label_only_training_set(param, X, y, clf_name, clf_method, noise_ratio=0.1, repeat=4, random_state=0,
                                          log_identifier=""):
    all_confusion_matrix = []
    prediction_per_sample = 2
    total_time = 0
    detection_errors = 0
    outlier_detection_confusion_matrix = []
    for i in range(repeat):
        model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            learning_rate=param["learning_rate"],
            n_estimators=param["n_estimators"],
            max_depth=param["max_depth"],
            num_leaves=param["num_leaves"],
            objective="binary"
        )
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1 / repeat * prediction_per_sample,
                                                            random_state=i)
        noised_train_y, groundtruth = add_noise_binary(train_y, noise_ratio, random_state=i)

        start = time.time()
        outlier_count = (len(groundtruth) - groundtruth.sum()) / 2
        print(clf_name, "start detecting", )

        if clf_name == "Local-Outlier-Factor" or clf_name == "Dummy":
            y_pred = clf_method.fit_predict(train_X)
            updated_y = update_by_outlier_prediction(train_y, y_pred)
        elif clf_name.startswith("3Sigma-"):
            y_pred = clf_method.fit_predict(train_X, train_y)
            updated_y = update_by_outlier_prediction(train_y, y_pred)
        else:
            clf_method.fit(train_X)
            y_pred = clf_method.predict(train_X)
            updated_y = update_by_outlier_prediction(train_y, y_pred)
        n_errors = (y_pred != groundtruth).sum()
        outlier_detection_confusion_matrix.append(confusion_matrix(groundtruth, y_pred))
        total_time = time.time() - start
        detection_errors += n_errors
        print(clf_name, "finish detecting", time.time() - start, n_errors)

        lgb_model = model.fit(train_X, updated_y)
        prediction = lgb_model.predict(test_X)
        conf_mat = confusion_matrix(test_y, prediction)
        all_confusion_matrix.append(conf_mat)

    aggregated_conf_mat = np.array(all_confusion_matrix).sum(axis=0) / prediction_per_sample
    file_log("active_modify_{}.log".format(log_identifier), noise_ratio, clf_name, total_time,
             aggregated_conf_mat, aggregated_conf_mat.sum(),
             np.array(outlier_detection_confusion_matrix).sum(axis=0) / prediction_per_sample, detection_errors)


if __name__ == '__main__':
    titanic_params = {
        'num_leaves': 20,
        'learning_rate': 0.1,
        'n_estimators': 120,
        'max_depth': 5
    }
    santander_params = {
        'max_depth': 5,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 40
    }
    X, y = load_safedriver()
    result = cv(X, y)
