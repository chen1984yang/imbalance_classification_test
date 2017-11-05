from dataset import *
from active_modify_sample_label import *
from outlier_detection import *

if __name__ == '__main__':
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

    step = 0.01
    stages = 6
    for i in range(0, stages):
        outliers_fraction = (i + 1) * step
        # outliers_fraction = 0
        detectors = generate_detectors(n_samples, n_features, outliers_fraction, random_state=i)
        # classifiers = generate_ksigma_detectors(n_features)
        for detector_name in detectors:
            # if clf_name != "Local-Outlier-Factor":
            #     continue
            detector = detectors[detector_name]
            active_modify_label_only_training_set(santander_params, X, y, detector_name, detector, noise_probability=outliers_fraction, threshold=0.13, log_identifier="santander_test_11", verbose=False, repeat=3)
