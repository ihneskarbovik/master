import test0_preparing_data as prep
import numpy as np
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

'''
Data preparation
'''

# Cleaning the data; dropping unnamed featues and setting target values to 1 or 0
X, y = prep.read_dataset('koding/datasets/test_sample_unique.csv')

# splitting the data into training and testing data
trainingset, testset = prep.split_time_series(X, y)

# choose a number of time steps
n_steps_in, n_steps_out = 10, 10
col_names = X.keys()

# plt.figure()
# for i, name in enumerate(col_names):
#     plt.subplot(len(col_names) + 1, 1, i + 1)
#     plt.plot(X[name])
#     plt.title(name, y=0.5, loc='right')
# plt.subplot(len(col_names) + 1, 1, len(col_names) + 1)
# plt.plot(y)
# plt.title('Mode', y=0.5, loc='right')
# plt.show()

'''
Building a "multiple input multi-step output" model
'''

# covert into input/output
X_train, y_train = prep.series_split_sequences(trainingset, n_steps_in, n_steps_out)
# the dataset knows the number of features
n_features = X_train.shape[2]
# reshape test set in same format as the training data
X_test, y_test = prep.series_split_sequences(testset, n_steps_in, n_steps_out)


losses = []
accuracies = []
for _ in range(20):
    # define model
    model = Sequential()
    model.add(LSTM(units=200, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
    model.add(LSTM(units=200, activation='relu'))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=400, verbose=0)
    # Final evaluation of the model
    y_pred = model.predict(X_test, verbose=0)
    y_pred = (y_pred > 0.5).astype(int)
    loss, acc = model.evaluate(X_test, y_test), prep.accuracy(y_pred,y_test)
    losses.append(loss)
    accuracies.append(acc)

print(f'Loss Mean: {np.mean(losses):.3f}')
print(f'Loss Standard Deviation: {np.std(losses):.3f}')
print(f'Accuracy Mean: {np.mean(accuracies):.3f}')
print(f'Accuracy Standard Deviation: {np.std(accuracies):.3f}')


'''
Building a "multiple parallel input and multi-step output" model
... tror jeg må bruke multiple input, ikke parallel ... 
'''

# # covert into input/output
# X_train, y_train = parallel_split_sequences(dataset, n_steps_in, n_steps_out)
# # the dataset knows the number of features
# n_features = X_train.shape[2]

# # define model
# model = Sequential()
# model.add(LSTM(units=200, activation='relu', input_shape=(n_steps_in, n_features)))
# model.add(RepeatVector(n_steps_out))
# model.add(LSTM(units=200, activation='relu', return_sequences=True))
# model.add(TimeDistributed(Dense(n_features)))
# model.compile(optimizer='adam', loss='mse')

# # fit the model
# model.fit(X_train, y_train, epochs=400, verbose=0)

# '''Evaluating the predictions on the test set'''

# # print(f'The accuracy is {accuracy(model.predict(X_test), y_test)}')