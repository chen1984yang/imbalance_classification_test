from dataset import *
from active_modify_sample_label import *
from outlier_detection import *
import os

if __name__ == '__main__':
    dataset = "santander"
    X, y = load_santander()
    # X, y = load_fraud_detection(sampled=False)
    print(X.shape)
    split = 1
    X = X[::split, :]
    y = y[::split]
    n_samples = len(y)
    n_features = X.shape[1]

    santander_params = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 120,
        'num_leaves': 40
    }

    columns = [
        "noise_ratio",
        "method",
        "time",
        "n_err",
        "accuracy",
        "classification true majority",
        "classification false minority",
        "classification false majority",
        "classification true minority",
        "detection true majority",
        "detection false minority",
        "detection false majority",
        "detection true minority"
    ]

    step = 0.01
    n_stage = 30
    k = 3

    log_file = "santander_test_11"

    column_names = ["noise_ratio", "method", "roc", "accuracy",
                    "classification true majority", "classification false minority",
                    "classification false majority", "classification true minority",
                    "detection true majority", "detection false minority",
                    "detection false majority", "detection true minority"
                    ]
    file_log(log_file, ",".join(column_names))
    all_result = pd.DataFrame(columns=column_names)

    noise_true_ratios = [0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.1]
    for i in range(18):
        noise_true_ratios.append(0.05 * (i + 1) + 0.1)
    # noise_true_ratios = []
    for i in range(10):
        noise_true_ratios.append(0.05 * (i + 1) + 1)
    noise_true_ratios.append(0.975)
    noise_true_ratios.append(1.025)

    noise_true_ratios = [0.25, 0.5, 0.75, 1.0, 1.25]

    # noise_rates = [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
    # noise_rates = [0.596]
    for i, noise_true_ratio in enumerate(noise_true_ratios):
        detectors = generate_detectors(n_samples, n_features, random_state=i)

        ksigma_detectors = generate_ksigma_detectors([1])
        detectors.update(ksigma_detectors)

        for detector_name in detectors:
            # if detector_name != "Robust-Covariance":
            # if detector_name != "Isolation-Forest":
            # if detector_name != "Dummy":
            if detector_name == "Dummy" or detector_name == "3Sigma-1":
                continue

            detector = detectors[detector_name]
            result = active_modify_label_only_training_set(santander_params, X, y, detector_name, detector,
                                                           noise_true_ratio=noise_true_ratio, threshold=0.5,
                                                           log_identifier=log_file, verbose=False, repeat=1, id=i,
                                                           random_state=i * 2, dataset=dataset)
            # print(result)
            all_result = all_result.append(pd.DataFrame([result], columns=column_names))

    print(all_result.describe())