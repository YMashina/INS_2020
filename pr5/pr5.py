from keras.models import Model, Sequential
from keras.layers import Input, Dense
from cmath import pi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import collections
import csv

def genData(size):
    data = np.zeros((size, 6))
    data_y = np.zeros(size)
    for i in range(size):
        x = x_sigma * np.random.randn(1) + x_mu
        e = y_sigma * np.random.randn(1) + y_mu
        data[i,:] = (x**2 + x + e, abs(x) + e, np.sin(x - pi/4)+e, np.log10(abs(x)) + e, -x**3+e, -x/4 + e)
        data_y[i] = -x + e
    mean_x = np.mean(data, axis=0)
    std_x = np.std(data, axis=0)
    data = (data - mean_x)/std_x
    mean_y = np.mean(data_y, axis=0)
    std_y = np.std(data_y, axis=0)
    data_y = (data_y - mean_y)/std_y
    return np.array(np.round(data, 3)), np.array(np.round(data_y, 3))


x_sigma = np.sqrt(10)
x_mu = 0
y_sigma = np.sqrt(0.3)
y_mu = 0

def save_csv(name, data):
    file = open(name, "w+")
    my_csv = csv.writer(file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    if isinstance(data, collections.Iterable) and isinstance(data[0], collections.Iterable):
        for i in data:
            my_csv.writerow(i)
    else:
        my_csv.writerow(data)


encode_input = Input(shape=(6,), name="encode_input")
enc = Dense(64, activation="relu")(encode_input)
enc = Dense(32, activation="relu")(enc)
enc = Dense(6, activation="linear")(enc)

dec_input = Input(shape=(6,), name="input_decoded")
dec = Dense(64, activation="relu")(dec_input)
dec = Dense(32, activation="relu")(dec)
dec = Dense(6, name="auto_aux")(dec)

pred = Dense(64, activation='relu')(enc)
pred = Dense(32, activation='relu')(pred)
pred = Dense(16, activation='relu')(pred)
pred = Dense(1, name="predict")(pred)

train_data, train_targets = genData(300)
test_data, test_targets = genData(100)

encoder = Model(encode_input, enc, name="encoder")
decoder = Model(dec_input, dec, name="decoder")
predicter = Model(encode_input, pred, name="predicter")
predicter.compile(optimizer="adam", loss="mse", metrics=["mae"])
a = predicter.fit(train_data, train_targets, epochs=100, batch_size=5, validation_split=0.2)

model = Sequential()
model.add(Dense(30, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse",metrics=['mae'])
H = model.fit(train_data, train_targets, epochs=100, batch_size=5, validation_split=0.2)

loss = a.history['loss']
model_loss = H.history['loss']
x = range(0, 100)
plt.plot(x, loss, 'b', label='my_loss')
plt.plot(x, model_loss, 'r', label='model_loss')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

my_train = a.history['val_loss']
model_train = H.history['val_loss']
plt.plot(x, my_train, 'b', label='my_val_loss')
plt.plot(x, model_train, 'r', label='model_val_loss')
plt.title('val_loss')
plt.ylabel('val_loss')
plt.xlabel('epochs')
plt.legend()
plt.show()
plt.clf()

save_csv('./train_data.csv', train_data)
save_csv('./train_targets.csv', train_targets)
save_csv('./test_data.csv', test_data)
save_csv('./test_targets.csv', test_targets)
save_csv('./encoded.csv', encoder.predict(test_data))
save_csv('./decoded.csv', decoder.predict(encoder.predict(test_data)))
save_csv('./regression.csv', predicter.predict(test_data))
decoder.save('decoder.h5')
encoder.save('encoder.h5')
predicter.save('predicter.h5')
