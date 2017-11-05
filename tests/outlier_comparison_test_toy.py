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
        'max_depth': 6,
        'n_estimators': 10,
        'num_leaves': 8
    }

    step = 0.03
    n_stage = 14
    for i in range(0, n_stage + 1):
        outliers_fraction = i * step
        # outliers_fraction = 0
        detectors = generate_detectors(n_samples, n_features, random_state=i)
        # classifiers = generate_ksigma_detectors(n_features)
        for detector_name in detectors:
            detector = detectors[detector_name]
            if detector_name != "Dummy":
                continue
            active_modify_label_only_training_set(toy_params, X, y, detector_name, detector, noise_probability=outliers_fraction, threshold=0.5, log_identifier="toy_test_11", verbose=False, repeat=1, id=i)
