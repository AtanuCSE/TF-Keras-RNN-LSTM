"""
# This code is for learning purposes
#
# This code is learned from the following tutorial
# https://www.udemy.com/course/complete-tensorflow-2-and-keras-deep-learning-bootcamp/
# It's been coded in PyCharm with the help of Anaconda
#
#  RNN for Sine Wave
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM

x = np.linspace(0, 50, 501)
y = np.sin(x)

#plt.plot(x,y)
#plt.show()

df = pd.DataFrame(data=y, index=x, columns=['sine'])

test_percentage = 0.1

test_point = np.round(len(df)*test_percentage)
test_ind = int(len(df) - test_point)  # Finding cuttoff point for train test split
train = df.iloc[:test_ind]
test = df.iloc[test_ind:]

# scaling / preprocessing
scaler = MinMaxScaler()
scaler.fit(train)
scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


length = 50
# How much input to see for the model to recognize any pattern
# The model will see 50 points and try to predict 51st point

batch_size = 1

generator = TimeseriesGenerator(scaled_train, scaled_train,
                                length=length, batch_size=batch_size)

# No of features
n_features=1

model = Sequential()
model.add(SimpleRNN(length, input_shape=(length,n_features)))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

print(model.summary())

# Training
h = model.fit_generator(generator, epochs=5)

# Model Evaluation
# Check losses
# loss vs validation_loss
loss_df = pd.DataFrame(h.history['loss'])
loss_df.plot()
plt.show()

# Now we need to take input points for testing
# The points required to generate next expected output
# For example, need to take values of 60 months to predict result of 61st month
# first_eval_batch = scaled_train[-length:]
# first_eval_batch = first_eval_batch.reshape((1, length, n_features))

# result = model.predict(first_eval_batch)
# print(result)

# Predict all test Data
test_prediction = []

first_eval_batch = scaled_train[-length:]
current_batch = first_eval_batch.reshape((1, length, n_features))

for i in range(len(test)):

    current_pred = model.predict(current_batch)[0]
    test_prediction.append(current_pred)
    # We need to add the new prediction to input set for generating new prediction
    current_batch = np.append(current_batch[:,1:,:], [[current_pred]], axis=1)

# Result is in test_prediction
# Lets compare with true results - scaled_test
# test_predictions are currently scaled
# Reverse it first
true_predictions = scaler.inverse_transform(test_prediction)
test['predictions'] = true_predictions
print(test)

test.plot(figsize=(12,8))
plt.show()
