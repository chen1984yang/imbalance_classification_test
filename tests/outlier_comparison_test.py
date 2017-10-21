from dataset import *
from active_modify_sample_label import *
from outlier_detection import *

if __name__ == '__main__':
    # X, y = load_santander()
    X, y = load_fraud_detection()
    print(X.shape)
    split = 100
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

    step = 0.05
    stages = 4
    for i in range(stages):
        outliers_fraction = (i + 1) * step
        classifiers = generate_detectors(n_samples, n_features, outliers_fraction)
        for clf_name in classifiers:
            clf = classifiers[clf_name]
            active_modify_label_only_training_set(fraud_params, X, y, clf_name, clf, outliers_fraction, log_identifier="fraud")
