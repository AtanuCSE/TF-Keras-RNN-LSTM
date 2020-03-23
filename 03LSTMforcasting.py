"""
# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
#
#  LSTM for Sine Wave
#  Use full data to train and forcast into the FUTURE
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

x = np.linspace(0, 50, 501)
y = np.sin(x)

#plt.plot(x,y)
#plt.show()

df = pd.DataFrame(data=y, index=x, columns=['sine'])

# scaling / preprocessing
full_scaler = MinMaxScaler()
scaled_full_data = full_scaler.fit_transform(df)

# ****************************************************************************
early_stop = EarlyStopping(monitor='loss', patience=2)

length = 50
# How much input to see for the model to recognize any pattern
# The model will see 50 points and try to predict 51st point

batch_size = 1

generator = TimeseriesGenerator(scaled_full_data, scaled_full_data,
                                length=length, batch_size=batch_size)

# No of features
n_features = 1

# Save multiple Training Time
if os.path.isfile('sine_prediction_lstm.h5'):
    model = load_model('sine_prediction_lstm.h5')
else:
    model = Sequential()
    model.add(LSTM(length, input_shape=(length, n_features)))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    print(model.summary())

    # Training
    h = model.fit_generator(generator, epochs=6,
                            callbacks=[early_stop])

    # Model Evaluation
    # Check losses
    # loss vs validation_loss
    loss_df = pd.DataFrame(h.history['loss'])
    loss_df.plot()
    plt.show()

    model.save('sine_prediction_lstm.h5')

# Forcast future values [not present in dataset]
forcast = []
no_of_prediction_point = 100  # Predict 100 future data

first_eval_batch = scaled_full_data[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(no_of_prediction_point):  # Predict 100 future data

    current_pred = model.predict(current_batch)[0]
    forcast.append(current_pred)
    # We need to add the new prediction to input set for generating new prediction
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

# Result is in forcast
# forcast are currently scaled
# Reverse it first
forcast = full_scaler.inverse_transform(forcast)

starting_prediction_index = 50 + 0.1
final_prediction_index = 50 + no_of_prediction_point * 0.1 + 0.1
print(final_prediction_index)
forcast_index = np.arange(starting_prediction_index, final_prediction_index, step=0.1)
print(forcast_index)
plt.plot(df.index, df['sine'])
plt.plot(forcast_index, forcast)
plt.show()














