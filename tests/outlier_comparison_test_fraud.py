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

    step = 0.0005
    n_stage = 4
    for i in range(1, n_stage + 1):
        outliers_fraction = (i) * step
        # outliers_fraction = 0
        detectors = generate_detectors(n_samples, n_features, random_state=i)
        # classifiers = generate_ksigma_detectors(n_features)
        for detector_name in detectors:
            detector = detectors[detector_name]
            active_modify_label_only_training_set(fraud_params, X, y, detector_name, detector, noise_probability=outliers_fraction, threshold=0.5, log_identifier="fraud_test_11", verbose=False, repeat=10)
