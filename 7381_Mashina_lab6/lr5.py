import string

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras import models
#from keras import layers
from tensorflow.keras.models import Sequential
from keras.regularizers import l2, l1
from keras.datasets import imdb
from tensorflow.keras.layers import Embedding, Dropout, Conv1D, MaxPool1D, GRU, LSTM, Dense, SimpleRNN, Flatten, Activation, LeakyReLU, PReLU
from keras.layers import Bidirectional, GlobalMaxPool1D
import tensorflow.keras.activations
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
# For adding new activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def swish(x):
    return (K.sigmoid(x) * x)

def vectorize(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1 # set specific indices of results[i] to 1s
    return results

def to_str(array):
    result = []
    for element in array:
        result.append(str(element))
    return result


def singleModelPlots( histories, dimensions = [10000] ):
    #print(histories.history)
    #plt.plot(histories.history['acc'])
    #plt.show()
    for history in histories:
        plt.plot(history.history['acc'])
    plt.title('Train accuracy with ' + activation )
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(to_str(dimensions), loc='upper left')
    plt.show()

    for history in histories:
        plt.plot(history.history['val_acc'])
    plt.title('Validation accuracy with ' + activation )
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(to_str(dimensions), loc='upper left')
    plt.show()

    for history in histories:
        plt.plot(history.history['loss'])
    plt.title('Train loss with ' + activation )
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(to_str(dimensions), loc='upper left')
    plt.show()

    for history in histories:
        plt.plot(history.history['val_loss'])
    plt.title('Validation loss with ' + activation )
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(to_str(dimensions), loc='upper left')
    plt.show()
    return


def build_model(dimension = 10000):
 #   model = Sequential()
  #  model.add(Embedding(input_dim=10000,output_dim=32,  input_length=10000))
    #model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    #model.add(MaxPool1D(pool_size=2))
    #model.add(LSTM(100))
    #model.add(GRU(32))
    #model.add(GRU(32, return_sequences=True))
    #model.add(SimpleRNN(16))
    #model.add(Dropout(0.2))
    #model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
    max_features = 9999
    embed_size = 256
    model = Sequential()

   # get_custom_objects().update({'swish': Activation(swish)})
    model.add(Dense(64, activation=activation, input_shape=(dimension,)))
    #model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=activation))
    #model.add(PReLU())
    model.add(Dropout(0.3))
    model.add(Dense(64, activation=activation))
    #model.add(PReLU())
    model.add(Dense(1, activation="sigmoid"))


    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model

def test_dimensions(dimensions=[10000]):
    results = []
    for dimension in dimensions:
        test_x, test_y, train_x, train_y = load_data(dimension)
        results.append(build_model(dimension).fit(
            train_x, train_y,
            epochs=epochs,
            batch_size=200,
            validation_data=(test_x, test_y)
        ))
    singleModelPlots(results, dimensions)
    return

def load_data(dimension=10000):
    (training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=dimension)
    data = np.concatenate((training_data, testing_data), axis=0)
    targets = np.concatenate((training_targets, testing_targets), axis=0)

    #print("Categories:", np.unique(targets))
    #print("Number of unique words:", len(np.unique(np.hstack(data))))

    length = [len(i) for i in data]

    #print("Average Review length:", np.mean(length))
    #print("Standard Deviation:", round(np.std(length)))
    #print("Label:", targets[0])
   # print(data[0])

    index = imdb.get_word_index()
    reverse_index = dict([(value, key) for (key, value) in index.items()])
    decoded = " ".join([reverse_index.get(i - 3, "#") for i in data[0]])

   # print(decoded)

    data = vectorize(data, dimension)
    targets = np.array(targets).astype("float32")


    test_x = data[:10000]
    test_y = targets[:10000]
    train_x = data[10000:]
    train_y = targets[10000:]
    return (test_x, test_y, train_x, train_y)

def test_my_text(filename, dimension=10000):

    text = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            text+=line.translate(str.maketrans('', '', string.punctuation)).lower().split()
    indexes = imdb.get_word_index() # use ready indexes
    print(indexes)
    print(text)
    encoded = []
    for word in text:
        if word in indexes and indexes[word] < 10000: # <10000 to avoid out of bounds error
            print('found '+word+' in indexes. its index is '+ str(indexes[word]))
            encoded.append(indexes[word])
    print('---------------------')
    print(np.array(encoded))

    reverse_index = dict([(value, key) for (key, value) in indexes.items()])

    decoded = " ".join([reverse_index.get(i , "#") for i in np.array(encoded)]) # не пон почему в ориге i-3
    print(decoded)
    test_x, test_y, train_x, train_y = load_data()

    print(decoded)
    #print(len(text.split()))
    model = build_model()
    model.fit(train_x, train_y, epochs=2, batch_size=200, validation_data=(test_x, test_y))
     # vectorize just like we did with data
    #print(model.predict(vectorize([np.array(encoded)])))
    return model.predict(vectorize([np.array(encoded)]))

activation = 'relu'
epochs = 2

if __name__ == '__main__':
    #test_dimensions([ 100, 1000, 5000, 10000, 20000, 30000, 40000])
    #test_dimensions() # for default launch on vector 10000
    #test_my_text('test.txt')
    print(str(test_my_text('test.txt'))+' '+str(test_my_text('test2.txt')))
   # print()