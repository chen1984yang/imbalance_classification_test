import numpy as np
import config

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

def list2str(list, delimiter=","):
    if type(list) == list:
        return delimiter.join([str(e) for e in list])
    elif type(list) == np.ndarray:
        if len(list.shape) > 1:
            list = np.ravel(list)
        return delimiter.join([str(e) for e in list])

    print("not a supported list.")

def normalize_data(a):
    col_mins = a.min(axis=0)
    col_maxs = a.max(axis=0)
    col_ranges = np.array(col_maxs - col_mins)
    for j in range(a.shape[1]):
        a[:, j] = (a[:, j] - col_mins[j]) / max(col_ranges[j], 1.)
        # for i in range(a.shape[0]):
        #     a[i][j] = (a[i][j] - col_mins[j]) / (col_ranges[j])
    # col_ranges = np.where(col_ranges <= 0., [1], col_ranges)
    # new_matrix = (a - col_mins) / col_ranges
    return a

def cluster2color(cluster_info):
    colors = []
    for i, j in enumerate(cluster_info):
        c = config.COLORS[cluster_info[i]]
        if j < 0:
            c += "0e"  # alpha channel
        colors.append(c)
    return np.array(colors)