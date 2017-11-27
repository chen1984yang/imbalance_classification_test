from dataset import *
from active_modify_sample_label import *
from outlier_detection import *

if __name__ == '__main__':
    # X, y = load_santander()
    X, y = load_fraud_detection(sampled=False)
    print(X.shape)
    split = 1
    print(Counter(y))
    X = X[::split, :]
    y = y[::split]
    n_samples = len(y)
    n_features = X.shape[1]

    fraud_params = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 100,
        'num_leaves': 30
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
    step = 0.005
    n_stage = 300
    log_file = "outlier-detection-fraud.log"
    file_log(log_file, ",".join(columns))
    if os.path.exists(log_file):
        os.remove(log_file)

    for i in range(0, n_stage + 1):
        noise_true_ratio = (i) * step
        # outliers_fraction = 0
        detectors = generate_detectors(n_samples, n_features, random_state=i)
        ksigma_detectors = generate_ksigma_detectors([1, 2, 5, 10, 15, 20, 25, 30])
        detectors.update(ksigma_detectors)

        for detector_name in detectors:
            detector = detectors[detector_name]
            active_modify_label_only_training_set(fraud_params, X, y, detector_name, detector, noise_true_ratio=noise_true_ratio, threshold=0.5, log_identifier=log_file, verbose=False, repeat=1)
