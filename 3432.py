import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model=Sequential([
    Dense(input_dim=2, units=1),
    Activation('sigmoid')
])
