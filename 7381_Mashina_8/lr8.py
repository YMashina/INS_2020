# Small LSTM Network to Generate Text for Alice in Wonderland
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
# load ascii text and covert to lowercase
filename = "wonderland.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)
# define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"

filepath_=filepath
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
print(checkpoint.best)


class MyCallback(Callback):
    def __init__(self):
        self.__epochCounter = 0

    def get_x(self):
        return self.__epochCounter

    def set_x(self, x):
        self.__epochCounter = x

    epochCounter = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epochCounter = self.epochCounter + 1
        if self.epochCounter == 1 or self.epochCounter % 5 == 0:
            print("End of raining epoch {}".format(epoch))
            epoch_predict(self.epochCounter, checkpoint.best)

callbacks_list = [
    checkpoint,
    MyCallback()
]

def epoch_predict(epoch, loss):
	import sys
	import numpy
	from keras.models import Sequential
	from keras.layers import Dense
	from keras.layers import Dropout
	from keras.layers import LSTM
	from keras.utils import np_utils
	# load ascii text and covert to lowercase
	filename = "wonderland.txt"
	raw_text = open(filename).read()
	raw_text = raw_text.lower()
	# create mapping of unique chars to integers, and a reverse mapping
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	# summarize the loaded data
	n_chars = len(raw_text)
	n_vocab = len(chars)
	print("Total Characters: ", n_chars)
	print("Total Vocab: ", n_vocab)
	# prepare the dataset of input to output pairs encoded as integers
	seq_length = 100
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	print("Total Patterns: ", n_patterns)
	# reshape X to be [samples, time steps, features]
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	# normalize
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	# define the LSTM model
	model = Sequential()
	model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	# load the network weights
	print(epoch, loss)

	filename = "weights-improvement-{epoch:02d}-{loss:.4f}.hdf5".format(epoch=epoch, loss=loss)
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# pick a random seed
	start = numpy.random.randint(0, len(dataX) - 1)
	pattern = dataX[start]
	print("Seed:")
	print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
	# generate characters
	generated_text = []
	for i in range(1000):
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(n_vocab)
		# print(x.shape)
		prediction = model.predict(x, verbose=0)
		index = numpy.argmax(prediction)

		result = int_to_char[index]
		generated_text.append(result)
		seq_in = [int_to_char[value] for value in pattern]
		sys.stdout.write(result)
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	print("\nDone.")
	f = open("generated.txt", "a")
	f.write('Epoch ' + str(epoch) + '\n' + ''.join(generated_text) + '\n')
	f.close()
	return


# fit the model
model.fit(X, y, epochs=30, batch_size=128, callbacks=callbacks_list)