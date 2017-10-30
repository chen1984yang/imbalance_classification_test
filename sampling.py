import numpy as np
from collections import Counter
import math
import random
import pandas as pd
from scipy.spatial.distance import euclidean

import tool

def reduce_majority(X, y, random_state=42, gamma=0.7):
    random.seed(random_state)
    stat = Counter(y)
    minority_class = min(stat, key=stat.get)
    n_minority = len(y[y == minority_class])

    ratio = math.pow(n_minority / len(y), gamma)
    print("sampling at", ratio)

    slices = y != y  # create a slice full of False values
    for i, l in enumerate(y):
        if l == minority_class:
            slices[i] = True
        if random.random() < ratio:
            slices[i] = True

    print("after sampling", len(slices[slices]))
    return X[slices], y[slices]

class DummySampler():
    def __init__(self):
        pass

    def fit_sample(self, X, y):
        return X, y

class KSigmaSampler():
    target_ratio = 0.1
    random_state = 0
    selection =np.array([])
    K = 3
    def __init__(self, target_ratio=1, random_state=0, K=3):
        """

        :param target_ratio: n_majority / n_minority
        :type target_ratio: float
        :param random_state:
        :type random_state:
        :param K:
        :type K:
        """
        self.target_ratio = target_ratio
        self.random_state = random_state
        self.K = K

    def fit_sample(self, X, y, ):
        random.seed(self.random_state)
        counter = Counter(y)
        minority_class = min(counter, key=counter.get)
        if type(X) == pd.DataFrame:
            X = X.values

        y = np.array(y)
        self.selection = y != y

        X = tool.normalize_data(X)
        mean_vector = np.mean(X, axis=0)
        minority = X[y == minority_class]
        majority = X[y != minority_class]

        distance = np.zeros_like(y)
        for i in range(len(y)):
            distance[i] = euclidean(X[i, :], mean_vector)

        all_mean = np.mean(distance)
        all_std = np.std(distance)

        n_minority = len(minority)
        majority_candidates = []

        for i, l in enumerate(y):
            if l == minority_class:
                self.selection[i] = True
                continue
            if abs(distance[i] - all_mean) > all_std * self.K:
                self.selection[i] = True
            else:
                majority_candidates.append(i)

        # calculate actual sampling ratio
        possible_min_ratio = max(1, len(y) - n_minority - len(majority_candidates)) / n_minority
        possible_max_ratio = (len(y) - n_minority) / n_minority
        if self.target_ratio < possible_min_ratio:
            self.target_ratio = possible_min_ratio
            print("up truncating target ratio to", possible_min_ratio)
        elif self.target_ratio > possible_max_ratio:
            self.target_ratio = possible_max_ratio
            print("down truncating target ratio to", possible_max_ratio)

        actual_ratio = (n_minority * self.target_ratio - len(y) + n_minority) / len(majority_candidates) + 1

        for l in majority_candidates:
            if random.random() < actual_ratio:
                self.selection[l] = True
            else:
                self.selection[l] = False

        sampled_X, sampled_y = X[self.selection], y[self.selection]
        n_sampled_minority = len(sampled_y[sampled_y == minority_class])
        n_sampled_majority = len(sampled_y[sampled_y != minority_class])
        print("KSigma sampling", self.target_ratio, actual_ratio, n_sampled_majority / n_sampled_minority, n_sampled_majority, n_sampled_minority)
        return sampled_X, sampled_y


def sampling_joparga3(data, X, y):
    # Number of data points in the minority class
    number_records_fraud = len(y[y == 1])
    fraud_indices = np.array(y[y == 1].index)

    # Picking the indices of the normal classes
    normal_indices = y[y == 0].index

    # Out of the indices we picked, randomly select "x" number (number_records_fraud)
    random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
    random_normal_indices = np.array(random_normal_indices)

    # Appending the 2 indices
    under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

    # Under sample dataset
    under_sample_data = data.iloc[under_sample_indices, :]

    X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
    y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']
    return X_undersample, y_undersample


from imblearn.under_sampling import *

def random_under_sampling(X, y, random_state=0):
    sampler = RandomUnderSampler(ratio='majority', random_state=random_state, return_indices=True)
    res = sampler.fit_sample(X, y)
    selected_index = res[1]
    return X.ix[selected_index], y.ix[selected_index]