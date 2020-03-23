"""
# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
#
#  RNN on Real Data / Time-stamped data
#  FRED data, clothing store, Validation
"""

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM
from tensorflow.keras.models import load_model

from tensorflow.keras.callbacks import EarlyStopping

# parse_date will automatically find and change date time object
df = pd.read_csv("RSCCASN.csv", parse_dates=True, index_col='DATE')  # Loading dataset

print(df.info())

df.columns = ['Sales']

df.plot()
plt.show()

# After observing data, it can be decided that 18 months can be test size
test_size = 18
test_ind = len(df) - test_size

train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

# scaling / preprocessing
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)

length = 12
batch_size = 1
generator = TimeseriesGenerator(scaled_train, scaled_train,
                                length=length, batch_size=batch_size)

# No of features
n_features=1

# Save multiple Training Time
if os.path.isfile('real_data_lstm.h5'):
    model = load_model('real_data_lstm.h5')
else:

    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    print(model.summary())

    early_stop = EarlyStopping(monitor='val_loss', patience=2)

    validation_generator = TimeseriesGenerator(scaled_test, scaled_test,
                                               length=length, batch_size=batch_size)

    # Training
    h = model.fit_generator(generator, epochs=20,
                            validation_data=validation_generator,
                            callbacks=[early_stop])

    # Model Evaluation
    # Check losses
    loss_df = pd.DataFrame(h.history['loss'])
    ax = loss_df.plot()
    loss_df2 = pd.DataFrame(h.history['val_loss'])
    loss_df2.plot(ax=ax)
    plt.show()

    model.save('real_data_lstm.h5')

# Predict all test Data
test_prediction = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):

    current_pred = model.predict(current_batch)[0]
    test_prediction.append(current_pred)
    # We need to add the new prediction to input set for generating new prediction
    current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

# Result is in test_prediction
# Lets compare with true results - scaled_test
# test_predictions are currently scaled
# Reverse it first
true_predictions = scaler.inverse_transform(test_prediction)
test['LSTM_predictions'] = true_predictions
print(test)

test.plot(figsize=(12, 8))
plt.show()

# ************* Forcast **********************************************
# scaling / preprocessing
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)

generator = TimeseriesGenerator(scaled_full_data, scaled_full_data,
                                length=length, batch_size=batch_size)

# No of features
n_features = 1

# Save multiple Training Time
if os.path.isfile('real_data_forcast_lstm.h5'):
    model = load_model('real_data_forcast_lstm.h5')
else:
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(length, n_features)))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Training
    h = model.fit_generator(generator, epochs=8)
    # no early_stopping coz no validation date while forcasting

    # Model Evaluation
    # Check losses
    # loss vs validation_loss
    loss_df = pd.DataFrame(h.history['loss'])
    loss_df.plot()
    plt.show()

    model.save('real_data_forcast_lstm.h5')

# Forcast future values [not present in dataset]
forcast = []
no_of_prediction_point = 12  # for 12 month

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(no_of_prediction_point):

    current_pred = model.predict(current_batch)[0]
    forcast.append(current_pred)
    # We need to add the new prediction to input set for generating new prediction
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

# Result is in forcast
# forcast are currently scaled
# Reverse it first
forcast = full_scaler.inverse_transform(forcast)

forcast_index = pd.date_range(start='2019-11-01', periods=no_of_prediction_point, freq='MS')
forcast_df = pd.DataFrame(data=forcast, index=forcast_index, columns=['Forcast'])

ax = df.plot(figsize=(12, 8))
forcast_df.plot(ax=ax)
plt.show()
