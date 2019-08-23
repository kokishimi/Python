# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 16:40:05 2019

@author: kokis
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn import datasets

np.random.seed()

mnist=datasets.fetch_mldata('MNIST original')
n=len(mnist.data)
N=30000
N_train=20000
N_validation=4000
indices=np.random.permutation(range(n))[:N]

X=mnist.data[indices]
X=X/255.0
X=X-X.mean(axis=1).reshape(len(X),1)
X=X.reshape(len(X),28,28)

y=mnist.target[indices]
Y=np.eye(10)[y.astype(int)]

X_train,X_test,Y_train,Y_test=\
    train_test_split(X,Y,train_size=N_train)

X_train,X_validation,Y_train,Y_validation=\
    train_test_split(X_train,Y_train,test_size=N_validation)
    
n_in=28
n_time=28
n_hidden=128
n_out=10

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

model=Sequential()
model.add(Bidirectional(LSTM(n_hidden),
                        input_shape=(n_time,n_in)))
model.add(Dense(n_out,init=weight_variable))
model.add(Activation('softmax'))

optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.9999)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

epochs=300
batch_size=250
early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)

hist=model.fit(X_train,Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation,Y_validation),
          callbacks=[early_stopping])

acc = hist.history['val_acc']
loss = hist.history['val_loss']

plt.rc('font', family='serif')
fig = plt.figure()
plt.plot(range(len(loss)), loss,
         label='loss', color='black')
plt.xlabel('epochs')
plt.show()

loss_and_metrics = model.evaluate(X_test, Y_test)
print(loss_and_metrics)