import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def read_dataset(filename):
    df = pd.read_csv(filename)
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    df = df.replace('DC', 1)
    df = df.replace('LTE', 0)

    X, y = df.drop('Mode', axis=1), df['Mode']
    return X, y

def split_time_series(X, y):
    tss = TimeSeriesSplit(n_splits=2)

    for train_index, test_index in tss.split(X):
        X_train, X_test = np.array(X.iloc[train_index, :]), np.array(X.iloc[test_index,:])
        y_train, y_test = np.array(y.iloc[train_index]), np.array(y.iloc[test_index])

    # reshaping y to match dimentions of X
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    # horizontally stack columns
    trainingset = np.hstack((X_train, y_train))
    testset = np.hstack((X_test, y_test))

    return trainingset, testset

def series_split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check to see if we are bwyond the data set
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix:out_end_ix, -1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def parallel_split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check to see if we are beyond the data set
        if out_end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def define_change(y):
    new_y = []
    for seq in y:
        if 0 in seq and 1 in seq:
            new_y.append(1)
        else:
            new_y.append(0)
    return new_y

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)
