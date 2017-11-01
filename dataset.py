from os.path import join
import pandas as pd
import numpy as np
import os
import time

location = os.path.dirname(os.path.abspath(__file__))
# print(location)

DATASET_ROOT = "data"

def load_toy() -> object:
    # load or create your dataset
    print('Load data...')
    df_train = pd.read_csv(join(location, DATASET_ROOT, 'multiclass_classification', 'multiclass.train'), header=None, sep='\t')
    df_test = pd.read_csv(join(location, DATASET_ROOT, 'multiclass_classification', 'multiclass.test'), header=None, sep='\t')

    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values
    return X_train, X_test, y_train, y_test


def load_titanic():
    # We will define a function to fill the missing entries in the age column
    def age_set(cols):
        age = cols[0]
        clas = cols[1]
        if pd.isnull(age):
            if clas == 1:
                return 37.0
            elif clas == 2:
                return 28.0
            else:
                return 24.0
        else:
            return age
    # load training data
    titanic_df = pd.read_csv(join(location, DATASET_ROOT, 'Titanic', 'train.csv'))
    titanic_df['Age'] = titanic_df[['Age', 'Pclass']].apply(age_set, axis=1)
    f_df = pd.get_dummies(titanic_df[['Embarked', 'Sex']], drop_first=True)
    titanic_df.drop(['Embarked', 'Sex'], axis=1, inplace=True)
    titanic_df = pd.concat([titanic_df, f_df], axis=1)
    X = titanic_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
    y = titanic_df['Survived']
    print(X.shape, y.shape)

    def set_fare(cols):
        pclass = cols[0]
        fare = cols[1]
        if pd.isnull(fare):
            if pclass == 1:
                return 1098.22
            elif pclass == 2:
                return 1117.94
            else:
                return 1094.17
        else:
            return fare

    # load testing data
    test_df = pd.read_csv(join(location, DATASET_ROOT, 'Titanic', 'test.csv'))
    test_df['Age'] = test_df[['Age', 'Pclass']].apply(age_set, axis=1)

    #Filling the empty Fare rows and dropping the Cabin column
    test_df['Fare'] = test_df[['Pclass','Fare']].apply(set_fare,axis=1)
    test_df.drop('Cabin',axis=1,inplace=True)
    test_df.dropna(axis=0,inplace=True)
    f1_df = pd.get_dummies(test_df[['Embarked', 'Sex']], drop_first=True)
    test_df.drop(['Embarked', 'Sex'], axis=1, inplace=True)
    test_df = pd.concat([test_df, f1_df], axis=1)
    test_data_id = test_df["PassengerId"]
    test_df1 = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    print(test_df1.shape)
    return (X, y, test_df1, test_data_id)

def load_santander():
    full_train = pd.read_csv(join(location, DATASET_ROOT, "Santander", "train.csv"), sep=",")
    # print(full_train.columns)
    y = full_train["TARGET"].values
    X = full_train.drop(labels=["ID", "TARGET"], axis=1).values
    print(full_train.shape)
    pos = y[y == 1]
    neg = y[y == 0]
    print(len(pos), len(neg))
    return X, y

def load_santander_test():
    test_X = pd.read_csv(join(location, DATASET_ROOT, "Santander", "test.csv"), sep=",")
    return test_X.values

def load_safedriver():
    full_train = pd.read_csv(join(location, DATASET_ROOT, "SafeDriver", "train.csv"), sep=",")
    # print(full_train.columns)
    y = full_train["target"].values
    X = full_train.drop(labels=["id", "target"], axis=1).values
    print(full_train.shape)
    pos = y[y == 1]
    neg = y[y == 0]
    print(len(pos), len(neg))
    return X, y

def load_fraud_detection(direct=True, sampled=False):
    if sampled:
        data = np.loadtxt(join(location, DATASET_ROOT, 'FraudDetection', 'creditcard-sampled.csv'), delimiter=',')
        return data[:, :-1], data[:, -1]
    start = time.time()
    data = np.loadtxt(join(location, DATASET_ROOT, 'FraudDetection', 'creditcard.csv'), delimiter=',', skiprows=1, dtype=bytes)

    data = np.array(data)
    full_X = data[:, 0:30].astype(float)
    labels = data[:, 30]
    # full_y = np.zeros(shape=(full_X.shape[0]))
    full_y = np.where(labels == b'"0"', [0], [1])
    print("fraud detection loaded in", time.time() - start, "s")
    return full_X, full_y

def load_synthetic_noise() -> (np.ndarray, np.ndarray, np.ndarray):
    """
    :return:
        merged_data: X values of synthesized data
        merged_y: y values of synthesized data, which contains noise, 0 means majority, 1 means minority
        outlier_groundtruth: indicates which instances are noise: -1 means noise, 1 mean normal
    :rtype:
    """
    minority_data = np.loadtxt(join(location, DATASET_ROOT, "synthetic_noise", "data_minority.csv"), delimiter=",")
    majority_data = np.loadtxt(join(location, DATASET_ROOT, "synthetic_noise", "data_majority.csv"), delimiter=",")
    noise_data = np.loadtxt(join(location, DATASET_ROOT, "synthetic_noise", "data_noise.csv"), delimiter=",")
    merged_data = np.concatenate((minority_data, majority_data, noise_data), axis=0)
    merged_y = np.array([1] * (len(minority_data) + len(noise_data)) + [0] * len(majority_data))
    outlier_groundtruth = np.array([1] * (len(majority_data) + len(minority_data)) + [-1] * len(noise_data))

    return merged_data, merged_y, outlier_groundtruth

if __name__ == '__main__':
    # X, y = load_fraud_detection(sampled=True)
    X, y, gnd = load_synthetic_noise()
    print(X.shape)
    from collections import Counter
    print(Counter(y))
    print(Counter(gnd))
    # import sampling
    # X_1, y_1 = sampling.reduce_majority(X, y)
    # pos = y_1[y_1 == 1]
    # neg = y_1[y_1 == 0]
    # np.savetxt(join(location, DATASET_ROOT, 'FraudDetection', 'creditcard-sampled.csv'), np.concatenate((X_1, y_1[:, np.newaxis]), axis=1), delimiter=',')
    # # print(X.shape)
    # print(y.shape)
    # load_titanic()
    # load_santander()
    # load_toy()
    # load_safedriver()
