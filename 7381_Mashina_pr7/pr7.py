from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np
import random
import math
import matplotlib.pyplot as plt

def func(i):
    i = i % 21
    return abs(i - 10)/4

def gen_sequence(seq_len = 1000):
    seq = [math.cos(i/4) + func(i) + random.normalvariate(0, 0.04) for i in range(seq_len)]
    return np.array(seq)

def draw_sequence():
    seq = gen_sequence(250)
    plt.plot(range(len(seq)),seq)
    plt.show()

def gen_data_from_sequence(seq_len=1006, lookback=10):
    seq = gen_sequence(seq_len)
    past = np.array([[[seq[j]] for j in range(i, i + lookback)] for i in range(len(seq) - lookback)])
    future = np.array([[seq[i]] for i in range(lookback, len(seq))])
    return (past, future)

data, res = gen_data_from_sequence()

dataset_size = len(data)
train_size = (dataset_size // 10) * 7
val_size = (dataset_size - train_size) // 2

train_data, train_res = data[:train_size], res[:train_size]
val_data, val_res = data[train_size:train_size + val_size], res[train_size:train_size + val_size]
test_data, test_res = data[train_size + val_size:], res[train_size + val_size:]

# LSTM’s and GRU’s were created as the solution to short-term memory
# сеть GRU - упрощенный вариант LSTM, обучается быстрее, архитектура проще,
# но не позволяет сохранять данные так долго, как может LSTM, поэтому подходит для решения более простых задач чем LSTM
model = Sequential()
#model.add(layers.GRU(64,recurrent_activation='sigmoid',input_shape=(None,1), return_sequences=True))
#model.add(layers.LSTM(64,activation='relu',input_shape=(None,1),return_sequences=True,dropout=0.3))
model.add(layers.GRU(32,recurrent_activation='relu',input_shape=(None,1), return_sequences=True))
model.add(layers.LSTM(32,activation='sigmoid',input_shape=(None,1),return_sequences=True,dropout=0.2))
model.add(layers.GRU(32,input_shape=(None,1),recurrent_dropout=0.1))
model.add(layers.Dense(1))

model.compile(optimizer='nadam', loss='mse')
history = model.fit(train_data, train_res, epochs=60, batch_size=8, validation_data=(val_data, val_res))
#print(history.history) # loss and val_loss
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss)
plt.plot(range(len(val_loss)), val_loss)
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

predicted_res = model.predict(test_data)
pred_length = range(len(predicted_res))
plt.plot(pred_length, predicted_res)
plt.plot(pred_length, test_res)
plt.title('Sequence prediction')
plt.legend(['Predict', 'Test'], loc='upper right')
plt.show()

#draw_sequence()