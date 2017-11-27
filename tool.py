import numpy as np
import config
from collections import Counter


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


def list2str(l, delimiter=","):
    if type(l) == list:
        return delimiter.join([str(e) for e in l])
    elif type(l) == np.ndarray:
        if len(l.shape) > 1:
            l = np.ravel(l)
        return delimiter.join([str(e) for e in l])

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

# def cluster2color(cluster_info):
#     colors = []
#     for i, j in enumerate(cluster_info):
#         c = config.COlORS[abs(j)]
#         if j < 0:
#             c += "ce"  # alpha channel
#         colors.append(c)
#     return np.array(colors)


def category2color(category, prev_color_map=None):
    n_data = len(category)
    stats = Counter(category)
    sorted_key = sorted(stats, key=stats.get)[::-1]
    
    if prev_color_map is None:
        prev_color_map = {}
    
    used_colors = list(prev_color_map.values())

    available_colors = list(config.COlORS[::-1])
    for c in used_colors:
        if c in available_colors:
            available_colors.remove(c)

    for i, k in enumerate(sorted_key):
        if k in prev_color_map:
            continue
        prev_color_map[k] = available_colors.pop()
        
    color_result = [None] * n_data
    for i, c in enumerate(category):
        color_result[i] = prev_color_map[c]
        
    return color_result, prev_color_map
