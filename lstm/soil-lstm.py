import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf

from torch.utils.data import TensorDataset, DataLoader

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# load the data
_dir = os.path.abspath('')
data_path = os.path.join(_dir, "../data/daily_cleaned.csv")
df = pd.read_csv(data_path)
df = df.drop(df.columns[0], axis=1)
new_columns = df.columns.values
new_columns[-1] = 'label'
df.columns = new_columns
#print(df)

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence: (t-n, ..., t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ..., t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # concatenate together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

values = df.values
data = series_to_supervised(values)
# drop the columns we don't want to predict (predicting for current time step), so all vars at time t except var5(t)
data.drop(data.columns[[5, 6, 7, 8]], axis=1, inplace=True)
#print(data)

# split dataset into train, validation, test sets
values = data.values
train_df = values[:1386, :] # train on the first 1386 days
valid_df = values[1386:2078, :] # validate on next 692 days
test_df = values[2078:, :] # test model on rest of the days

# setup train data
train_x, train_y = train_df[:, :-1], train_df[:, -1] # raw numpy

# setup validation data
valid_x, valid_y = valid_df[:, :-1], valid_df[:, -1]

# setup test data
test_x, test_y = test_df[:, :-1], test_df[:, -1]

# reshape inputs (x's) to be 3D [seq_len, batch, input_size]
# using batches of 30, sequence length should always be 1
# should be (30, 1, m)
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
valid_x = valid_x.reshape((valid_x.shape[0], 1, valid_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)

# create keras model
model = Sequential()
model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(1))

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='mae', optimizer=opt, metrics=['accuracy'])
# fit the model
history = model.fit(train_x, train_y, epochs=100, batch_size=30, validation_data=(valid_x, valid_y), verbose=2, shuffle=False)
# plot the history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

# evaluate the model on test set
pred = model.evaluate(test_x, test_y)
print(pred)