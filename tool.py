import numpy as np

def count_zero_sample(X):
    abs_X = abs(X)
    feature_sum = np.sum(abs_X, 1)
    # print(feature_sum.shape)
    zero_count = len(feature_sum[feature_sum == 0])
    return zero_count


def get_histogram(arr, bins=100, proba=False):
    hist, bins = np.histogram(arr, bins=bins)

    if not proba:
        return hist, bins

    sum = len(arr)
    prob_distribution = np.zeros_like(hist, dtype=np.float16)
    for i, e in enumerate(hist):
        if e == 0:
            prob_distribution[i] = 1e-6
        else:
            prob_distribution[i] = e / sum
    return prob_distribution, bins
