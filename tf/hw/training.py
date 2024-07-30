from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid

hidden_neurons = 30

model = Sequential()
model.add(Input(shape=(28, 28)))
model.add(Flatten())
model.add(Dense(units=hidden_neurons, activation=sigmoid))
model.add(Dense(units=hidden_neurons, activation=sigmoid))
model.add(Dense(units=10, activation=sigmoid))

model.compile