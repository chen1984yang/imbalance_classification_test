import random

from dataset import *
from logger import *
from training import *
import tool
import outlier_detection
import log_io
import simple_plot
from sklearn.neighbors import LocalOutlierFactor


def modify_label_dummy(X, y):
    name = "dummy"
    return y, np.ones_like(y)


from collections import Counter


def add_noise_to_majority(y, target_count, random_state=0, verbose=False):
    noised_y = np.array(y)
    groundtruth = np.ones_like(y)
    random.seed(random_state * 2)

    counter = Counter(y)
    minority_class = min(counter, key=counter.get)
    # print('add noise: minority label', minority_class)
    class_labels = list(counter.keys())
    class_labels.remove(minority_class)

    outlier_count = 0
    majority_indices = []
    # minority_indices = []
    for i, label in enumerate(y):
        if label == minority_class:
            continue
        majority_indices.append(i)
    np.random.seed(random_state)
    sampled_majority_indices = np.random.choice(majority_indices, target_count, replace=False)
    outlier_count = len(sampled_majority_indices)
    for i, index in enumerate(sampled_majority_indices):
        noised_y[index] = minority_class
        groundtruth[index] = -1

    if verbose:
        print(outlier_count, "outlier added")
    return noised_y, groundtruth


from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score


def update_by_outlier_prediction(original_y, outlier_prediction) -> np.ndarray:
    updated_y = np.zeros_like(original_y)
    # print("original", Counter(original_y))
    for i, l in enumerate(outlier_prediction):
        if l == -1:
            updated_y[i] = 1 - original_y[i]
        else:
            updated_y[i] = original_y[i]
    # print("updated", Counter(updated_y))
    return updated_y

from sklearn.metrics import roc_auc_score

def active_modify_label_only_training_set(param, X, y, detector_name, detector, noise_true_ratio=0.1, threshold=0.5, repeat=10, random_state=0, log_identifier="", verbose=False, id=0, dataset='toy'):
    all_confusion_matrix = []
    total_time = 0
    detection_errors = 0
    n_instances, n_features = X.shape
    outlier_detection_confusion_matrix = []
    counter = Counter(y)
    minority_class = min(counter, key=counter.get)
    roc = []
    noise_proportition = 0
    accs = []
    added_noise_count = 0
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

        # model = RandomForestClassifier(
        #     n_estimators=param["n_estimators"],
        #     max_depth=param["max_depth"],
        #     random_state=random_state
        # )

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2,
                                                            random_state=i)
        true_minority_indices = np.array(train_y == minority_class)
        current_noise_count = int(noise_true_ratio * true_minority_indices.sum())
        noised_train_y, groundtruth = add_noise_to_majority(train_y, current_noise_count, random_state=i, verbose=verbose)

        added_noise_count += Counter(groundtruth)[-1]

        noised_minority_indices = noised_train_y==minority_class
        noise_proportition = 1 - true_minority_indices.sum() / noised_minority_indices.sum()
        current_noise_true_ratio = round(noise_proportition / (1 - noise_proportition), 3)

        if detector.__class__ == LocalOutlierFactor:
            detector = LocalOutlierFactor(n_neighbors=5, contamination=min(noise_proportition, 0.5))
        detector.set_params(contamination=noise_proportition)
        noised_minority_X = train_X[noised_minority_indices]
        if noise_proportition > 0:
            start = time.time()
            if verbose:
                print(detector_name, "start detecting")

            # predict and update
            outlier_prediction = outlier_detection.omni_detector_detect(detector, noised_minority_X)

            expanded_outlier_prediction = np.ones(shape=(len(train_y)))
            expanded_outlier_prediction[noised_minority_indices] = outlier_prediction
            updated_y = update_by_outlier_prediction(noised_train_y, expanded_outlier_prediction)

            # collect stats info
            n_errors = (outlier_prediction != groundtruth[noised_minority_indices]).sum()
            detection_cf = confusion_matrix(groundtruth[noised_minority_indices], outlier_prediction)
            outlier_detection_confusion_matrix.append(np.ravel(detection_cf))
            total_time = time.time() - start
            detection_errors += n_errors
            if verbose:
                print(detector_name, "finish detecting", time.time() - start, n_errors)
        else:
            updated_y = train_y
            expanded_outlier_prediction = np.ones_like(train_y)
            detection_cf = confusion_matrix(groundtruth[noised_minority_indices], groundtruth[noised_minority_indices])
            outlier_detection_confusion_matrix.append([0, 0, 0, detection_cf[0][0]])

        trial_id = "{}-{}-{:.3f}".format(detector_name, id, noise_true_ratio)

        # train model
        lgb_model = model.fit(train_X, updated_y)

        # predict training set
        prediction_training = lgb_model.predict(train_X)

        # # save plot: based on updated_y
        # training_predicted_result = updated_y + prediction_training * 2
        # colors, _ = tool.category2color(training_predicted_result, {
        #     0: "#5079a5",
        #     1: "#dd565c",
        #     2: "#79b7b2",
        #     3: "#ef8e3b"
        # })
        # trial_id = "{}-{}-{:.4f}".format(detector_name, id, noise_true_ratio)
        # # save plots
        # path = join("figures", "gaussian-using-noised-label", trial_id + ".png")
        # simple_plot.save_plot2png(train_X[:, 0], train_X[:, 1], colors, path, noise_true_ratio)

        # # save plot: based on updated_y
        # training_predicted_result = train_y + prediction_training * 2
        # colors, _ = tool.category2color(training_predicted_result, {
        #     0: "#5079a5",
        #     1: "#dd565c",
        #     2: "#79b7b2",
        #     3: "#ef8e3b"
        # })
        # # save plots
        # folder = join("figures", "gaussian-using-true-label", detector_name)
        # if not os.path.exists(folder):
        #     os.mkdir(folder)
        # path = join(folder, trial_id + ".png")
        # simple_plot.save_plot2png(train_X[:, 0], train_X[:, 1], colors, path, noise_true_ratio)

        # save detection and classification result
        all_info = np.concatenate((train_X,
                train_y[:, np.newaxis],
                groundtruth[:, np.newaxis],
                expanded_outlier_prediction[:, np.newaxis],
                prediction_training[:, np.newaxis]
            ),
            axis=1
        )
        result_root = os.path.join("outlier-result", log_identifier)
        if not os.path.exists(result_root):
            os.mkdir(result_root)
        np.savetxt(os.path.join(result_root, trial_id + ".csv"), all_info, delimiter=',', fmt='%d')

        # predict testing set
        predicted_proba = lgb_model.predict_proba(test_X)
        prediction_proba = predicted_proba[:, 1]
        prediction = np.where(prediction_proba > threshold, [1], [0])

        # metrics
        auc = roc_auc_score(test_y, prediction_proba)
        acc = accuracy_score(test_y, prediction)
        accs.append(acc)
        roc.append(auc)
        conf_mat = confusion_matrix(test_y, prediction)
        conf_mat = np.ravel(conf_mat)
        all_confusion_matrix.append(conf_mat)

    # aggregate metric results
    aggregated_conf_mat = np.array(all_confusion_matrix).mean(axis=0)
    aggregated_detection_conf_mat = np.array(outlier_detection_confusion_matrix).mean(axis=0)

    # current_noise_count = noise_proportition / (1 - noise_proportition)
    current_noise_true_ratio = noise_true_ratio
    roc_mean = np.array(roc).mean()
    acc_mean = np.array(accs).mean()
    print("average noise count", added_noise_count / repeat)
    majority_class = 1 - minority_class
    classification_conf_mat = np.array(aggregated_conf_mat).ravel().tolist()
    detection_conf_mat = np.array(aggregated_detection_conf_mat).ravel().tolist()
    metric_bundle = [current_noise_true_ratio, detector_name, roc_mean, acc_mean] + classification_conf_mat + detection_conf_mat

    file_log(log_identifier, *metric_bundle)
    return metric_bundle


def parse_log_result(data: {}, index_key, filter_key, row_index):
    method_names = set(data[index_key])

    n_data = len(data[index_key])
    result = {}
    x_values = set(data[row_index])
    x_values = sorted(x_values)
    length_for_series = []
    for name in method_names:
        l = []
        for i in range(n_data):
            if data[index_key][i] == name:
                l.append((data[filter_key][i], data[row_index][i]))

        l = sorted(l, key=lambda e: e[1])
        result[name] = [e[0] for e in l]
        length_for_series.append(len(result[name]))
    # print(result)
    min_length = np.min(length_for_series)
    for name in result:
        result[name] = result[name][:min_length]
    x_values = x_values[:min_length]
    return result, x_values


if __name__ == '__main__':
    # dataset = "santander"
    dataset = "toy"
    # dataset = "fraud"
    path = os.path.join("tests", "{}_test_11".format(dataset))
    r = log_io.from_csv(path)
    column_names = ["accuracy",
                    "classification true majority",
                    "classification false minority",
                    "classification false majority",
                    "classification true minority",
                    "detection true majority",
                    "detection false minority",
                    "detection false majority",
                    "detection true minority"
                    ]

    x_column = "noise_ratio"
    x_title = "noise / true minority"
    color_map = {
        "Dummy": "#949494",
        "Isolation-Forest": "#1E76B4",
        "3Sigma": "#FF7F0E",
        "Local-Outlier-Factor": "#D62728",
        "Robust-Covariance": "#9467BD"
    }
    for column in column_names:
        result_title = "result-{}-{}".format(dataset, column)
        result, x_values = parse_log_result(r, "method", column, x_column)
        simple_plot.save_linechart(result, x_values, result_title + ".png", title=result_title, x_title=x_title,
                                   y_title=column, color_map=color_map)

