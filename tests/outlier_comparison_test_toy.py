from dataset import *
from active_modify_sample_label import *
from outlier_detection import *

if __name__ == '__main__':
    # X, y = load_santander()
    X, y, gnd = load_synthetic_noise_gaussian()
    # print(Counter(y))
    y -= np.array((1 - gnd) / 2).astype(np.int64)
    print(Counter(y))
    print(X.shape)
    split = 1
    X = X[::split, :]
    y = y[::split]
    n_samples = len(y)
    n_features = X.shape[1]

    toy_params = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 20,
        'num_leaves': 5
    }

    step = 0.03
    n_stage = 25

    k = 3

    column_names = ["noise_ratio", "method", "roc", "accuracy",
        "classification true majority", "classification false minority",
        "classification false majority", "classification true minority",
        "detection true majority", "detection false minority",
        "detection false majority", "detection true minority"
    ]
    all_result = pd.DataFrame(columns=column_names)

    for i in range(0, n_stage + 1):
        outliers_fraction = i * step
        # outliers_fraction = 0
        # detectors = generate_detectors(n_samples, n_features, random_state=i)
        detectors = generate_ksigma_detectors(n_features, k)
        for detector_name in detectors:
            detector = detectors[detector_name]
            # if detector_name != "Local-Outlier-Factor":
            # if detector_name != "Dummy":
            if detector_name != "{}Sigma-1".format(k):
                continue
            result = active_modify_label_only_training_set(toy_params, X, y, detector_name, detector, noise_probability=outliers_fraction, threshold=0.5, log_identifier="toy_test_11", verbose=False, repeat=10, id=i)
            print(result)
            all_result = all_result.append(pd.DataFrame([result], columns=column_names))

    print(all_result.describe())

