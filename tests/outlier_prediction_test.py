from outlier_detection import *
import sampling
import tsne_plot
import tool
import dataset
import sys
from scipy import stats

if __name__ == '__main__':
    seed = 42
    n_split = 10
    # noise_rates = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    noise_rates = [0.04, 0.4]
    dataset_name = "fraud"
    file_name = "cv-tsne-result-fraud-0.0134-3330-492.npy"

    # dataset_name = "santander"
    # file_name = "cv-tsne-result-santander-0.7335-52751-3008.npy"

    path = join(file_name)
    cached_result = np.load(path)

    # # add original feature
    # X, y = dataset.load_santander()
    # sampled_X, sampled_y = sampling.reduce_majority(X, y, gamma=0.1)
    # cached_result = np.concatenate((sampled_X, sampled_y[:, np.newaxis], cached_result), axis=1)
    # np.save(file_name, cached_result)
    # sys.exit(0)

    X, y, tsne_result = tsne_plot.parse_cv_tsne_result(cached_result, n_split=n_split)
    n_samples, n_features = X.shape
    counter = Counter(y)
    print(counter)
    minority_class = min(counter, key=counter.get)
    print(X.shape, y.shape)

    np.save(file_name, cached_result)

    for i, r in zip(range(n_split), tsne_result):
        corr, selection = r
        corr_x = corr[:, 0]
        corr_y = corr[:, 1]
        print(np.histogram(corr_x))
        print(np.histogram(corr_y))
        train_X = X[selection == 0]
        train_y = y[selection == 0]

        for rate in noise_rates:
            detectors = generate_detectors(n_samples, n_features, rate, random_state=seed)
            noised_train_y, groundtruth = add_noise_to_majority(train_y, rate, random_state=seed, verbose=100)
            groundtruth = np.where(groundtruth == 1, [1], [0])  # 1 means normal, 0 means outlier
            noised_minority_indices = noised_train_y == minority_class
            noised_minority_X = train_X[noised_minority_indices]
            noised_minority_y = train_y[noised_minority_indices]

            for detector_name in detectors:
                # if detector_name.startswith("3Sigma"):
                #     continue
                if rate != 0:
                    outlier_prediction = omni_detector_detect(detectors[detector_name], noised_minority_X)
                    outlier_prediction = np.where(outlier_prediction == 1, [1], [-1])
                else:
                    outlier_prediction = np.ones(shape=(len(noised_minority_X)), dtype=np.int)

                detection_confusion_matrix = confusion_matrix(groundtruth[noised_minority_indices], outlier_prediction)
                print(np.ravel(detection_confusion_matrix))
                color_index = - np.zeros(shape=len(train_y), dtype=np.int)
                prediction_for_noised_only = []
                for idx, instance_outlier_groundtruth, instance_outlier_prediction, instance_true_label in zip(
                        range(len(noised_minority_y)),
                        groundtruth[noised_minority_indices],
                        outlier_prediction,
                        train_y[noised_minority_indices]):
                    color_type = -1
                    if instance_outlier_groundtruth == instance_outlier_prediction:  # correct prediction
                        if instance_true_label == minority_class:
                            color_type = 2  # "#dc3912" orange TN
                        else:
                            color_type = 1  # "#3366cc" blue TP
                    else:
                        if instance_true_label == minority_class:
                            color_type = 3  # green FN
                        else:
                            color_type = 4  # violet FP
                    prediction_for_noised_only.append(color_type)

                color_index[noised_minority_indices] = prediction_for_noised_only
                print(detector_name, rate, Counter(color_index))
                # colors = - np.ones(shape=(len(train_y)), dtype=np.int)
                # colors = prediction_result
                colors = tool.cluster2color(color_index)

                figure_name = "{}-{:.2f}-{}-{}.png".format(dataset_name, rate, i, detector_name)
                path = join(config.FIGURE_ROOT, figure_name)
                tsne_plot.save_plot2png(corr_x[selection==0], corr_y[selection==0], colors, path)
                print("saved to", path)

        break

    # X, y = load_fraud_detection()
    #
    # n_samples = len(y)
    # n_features = X.shape[1]
    # random.seed(seed)
    #
    # sampled_X, sampled_y = reduce_majority(X, y, gamma=0.99)
    # sampling_ratio = len(sampled_y) / len(y)
    # c = Counter(sampled_y)
    # print(c)
    #
    #