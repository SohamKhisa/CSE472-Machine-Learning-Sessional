import numpy as np
import pandas as pd
import sys

def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    # todo: implement
    if len(sys.argv) < 2:
        print('No file name provided')
        sys.exit(1)
    file = sys.argv[1]

    df = pd.read_csv(file)
    df = df.dropna()
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # print('The dataset x is: \n', X.head())
    # print('The dataset X shuffled is: \n', X.sample(frac=1).head())
    # print('The dataset y is: \n', y.head())
    return X, y


def split_dataset(X, y, test_size=0.2, shuffle=True):
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.
    X_train, y_train, X_test, y_test = None, None, None, None
    if test_size >= 1 or test_size <= 0:
        raise ValueError('test_size must be between 0 and 1 (exc1usive)')
    length = len(X)
    n_test = int(np.ceil(length*test_size))
    n_train = length - n_test

    if shuffle:
        # perm is a random list of numbers from 0 to length
        random_seed=5
        perm = np.random.permutation(length)
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        train_indices = np.arange(n_train)
        test_indices = np.arange(n_train, length)
    
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # converting dataframe to numpy array and also getting rid of rank 1 array, (n,) -> (n, 1)
    # X_train = X_train.to_numpy().reshape(n_train, X_train.shape[1])
    # X_test = X_test.to_numpy().reshape(n_test, X_test.shape[1])
    # y_train = y_train.to_numpy().reshape(n_train, 1)
    # y_test = y_test.to_numpy().reshape(n_test, 1)

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    X_sample, y_sample = None, None
    sample = len(X)
    # generating indices of random sample with replacement
    perm = np.random.choice(sample, size=sample, replace=True)
    X_sample = X.iloc[perm]
    y_sample = y.iloc[perm]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
