import random

from dataset import *
from logger import *
from training import *
from tool import *
import outlier_detection


def modify_label_dummy(X, y):
    name = "dummy"
    return y, np.ones_like(y)


from collections import Counter


def add_noise_to_majority(y, noise_ratio, random_state=0, verbose=False):
    noised_y = np.array(y)
    groundtruth = np.ones_like(y)
    random.seed(random_state * 2)

    counter = Counter(y)
    minority_class = min(counter, key=counter.get)
    class_labels = list(counter.keys())
    class_labels.remove(minority_class)

    outlier_count = 0
    for i, label in enumerate(y):
        if label == minority_class:
            continue
        if random.random() < noise_ratio:
            noised_y[i] = minority_class
            groundtruth[i] = - 1
            outlier_count += 1
    if verbose:
        print(outlier_count, "outlier added")
    return noised_y, groundtruth


from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix


def update_by_outlier_prediction(original_y, outlier_prediction) -> np.ndarray:
    updated_y = np.zeros_like(original_y)
    for i, l in enumerate(outlier_prediction):
        if l == -1:
            updated_y[i] = 1 - original_y[i]
        else:
            updated_y[i] = original_y[i]
    return updated_y

from sklearn.metrics import roc_auc_score

def active_modify_label_only_training_set(param, X, y, detector_name, detector, noise_probability=0.1, threshold=0.5, repeat=10, random_state=0, log_identifier="", verbose=False):
    all_confusion_matrix = []
    prediction_per_sample = 2
    total_time = 0
    detection_errors = 0
    n_instances, n_features = X.shape
    outlier_detection_confusion_matrix = []
    counter = Counter(y)
    minority_class = min(counter, key=counter.get)
    roc = []
    noise_proportition = 0
    for i in range(repeat):
        model = lgb.LGBMClassifier(
            boosting_type="gbdt",
            learning_rate=param["learning_rate"],
            n_estimators=param["n_estimators"],
            max_depth=param["max_depth"],
            num_leaves=param["num_leaves"],
            objective="binary",
            seed=random_state
        )
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=1 / repeat * prediction_per_sample,
                                                            random_state=i)
        true_minority_indices = np.array(train_y == minority_class)
        noised_train_y, groundtruth = add_noise_to_majority(train_y, noise_probability, random_state=i, verbose=verbose)

        noised_minority_indices = noised_train_y==minority_class
        noise_proportition = 1 - true_minority_indices.sum() / noised_minority_indices.sum()
        print("contamination", noise_proportition)
        detector.set_params(contamination=noise_proportition)
        noised_minority_X = train_X[noised_minority_indices]
        if noise_probability > 0:
            start = time.time()
            if verbose:
                print(detector_name, "start detecting")

            # predict and update
            outlier_prediction = outlier_detection.omni_detector_detect(detector, noised_minority_X)
            expanded_outlier_prediction = np.ones(shape=(len(train_y)))
            expanded_outlier_prediction[noised_minority_indices] = outlier_prediction
            updated_y = update_by_outlier_prediction(train_y, outlier_prediction)

            # collect stats info
            n_errors = (outlier_prediction != groundtruth[noised_minority_indices]).sum()
            outlier_detection_confusion_matrix.append(confusion_matrix(groundtruth[noised_minority_indices], outlier_prediction))
            total_time = time.time() - start
            detection_errors += n_errors
            if verbose:
                print(detector_name, "finish detecting", time.time() - start, n_errors)
        else:
            updated_y = train_y
            outlier_detection_confusion_matrix = [[], []]

        lgb_model = model.fit(train_X, updated_y)
        predicted_proba = lgb_model.predict_proba(test_X)
        prediction = predicted_proba[:, 1]
        prediction = np.where(prediction > threshold, [1], [0])
        auc = roc_auc_score(test_y, prediction)
        roc.append(auc)
        conf_mat = confusion_matrix(test_y, prediction)
        all_confusion_matrix.append(conf_mat)

    aggregated_conf_mat = np.array(all_confusion_matrix).sum(axis=0) / prediction_per_sample
    aggregated_detection_conf_mat = np.array(outlier_detection_confusion_matrix).sum(axis=0) / prediction_per_sample
    print(aggregated_detection_conf_mat, aggregated_conf_mat)

    file_log("active_modify_{}.log".format(log_identifier),
             noise_proportition, detector_name,
             total_time, np.array(roc).mean(), detection_errors,
             list2str(aggregated_conf_mat),
             list2str(aggregated_detection_conf_mat))


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
