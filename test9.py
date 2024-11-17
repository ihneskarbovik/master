# forsøk på debugging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from time import time


def read_dataset(filename, campaign=None):
    df = pd.read_csv(filename)
    df = df.loc[:, ~df.columns.str.match('Unnamed')]
    df = df.loc[:, ~df.columns.str.match('Timestamp')]
    df = df.replace('DC', 1)
    df = df.replace('LTE', 0)
    df = df[df['campaign'].str.contains('Driving')]

    if campaign != None:
        df = df[df['campaign'] == campaign]

    X, y = df[['RSRP', 'SSS_RSRP', 'campaign', 'Mode']], df['Mode']
    return X, y


def series_split_sequences(f, t, n_steps_in, n_steps_out):
    X, y = [], []
    curr_campaign = ''
    for i in range(len(f)):
        # find the end of the pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(f): # check to see if we are bwyond the data set
            break

        if curr_campaign == '': # set new current campaign if there is np campaign
            curr_campaign = f['campaign'].iloc[i]
        seq_x, seq_y = f[i:end_ix], t[end_ix:out_end_ix]
        if f.iloc[i:out_end_ix]['campaign'].nunique() > 1: # check to see if in transition between two campaigns
            continue
        elif curr_campaign != f['campaign'].iloc[i]: # set new campaign equal to current if this is the campaign we are looping through
            curr_campaign = f['campaign'].iloc[i]
        X.append(seq_x.drop('campaign', axis=1))
        y.append(seq_y)
    return np.array(X), np.array(y)


def vertical_handover(y):
    new_y = []
    for seq in y:
        if 0 in seq and 1 in seq:
            new_y.append(1)
        else:
            new_y.append(0)
    return np.array(new_y)


campaign='Interactivity_gaming_4G5G_TIM_Driving_Palasport_to_Via_Appia_End_No_Flight_Mode'
X, y = read_dataset('datasets/Op1_merged.csv', campaign=campaign)

new_y = []
for i in range(len(y) - 1):
    if y.iloc[i] == y.iloc[i+1]:
        new_y.append(0)
    else:
        new_y.append(1)

print(np.sum(new_y))
# n_steps_in, n_steps_out = 15, 1
# X, y = series_split_sequences(X, y, n_steps_in, n_steps_out)