# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:29:40 2019

@author: kokis
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import GRU
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

np.random.seed()

def mask(T=200):
    mask=np.zeros(T)
    indices=np.random.permutation(np.arange(T))[:2]
    mask[indices]=1
    return mask

def toy_problem(N=10, T=200):
    signals=np.random.uniform(low=0.0,high=1.0,size=(N,T))
    masks=np.zeros((N,T))
    for i in range(N):
        masks[i]=mask(T)
        
    data=np.zeros((N,T,2))
    data[:,:,0]=signals[:]
    data[:,:,1]=masks[:]
    target=(signals*masks).sum(axis=1).reshape(N,1)
    
    return (data,target)

N=10000
T=200
maxlen=T

X,Y=toy_problem(N=N,T=T)

N_train=int(N*0.9)
N_validation=N-N_train

X_train,X_validation,Y_train,Y_validation=\
    train_test_split(X,Y,test_size=N_validation)
    
n_in=len(X[0][0])
n_hidden=100
n_out=len(Y[0])

def weight_variable(shape, name=None):
    return np.random.normal(scale=.01, size=shape)

model=Sequential()
model.add(GRU(n_hidden,
               kernel_initializer=weight_variable,
               input_shape=(maxlen,n_in)))
model.add(Dense(n_out,kernel_initializer=weight_variable))
model.add(Activation('linear'))

optimizer=Adam(lr=0.001,beta_1=0.9,beta_2=0.9999)
model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['mae'])

epochs=500
batch_size=100
early_stopping=EarlyStopping(monitor='val_loss',patience=100,verbose=1)

hist=model.fit(X_train,Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_validation,Y_validation),
          callbacks=[early_stopping])

Z=X[:1] #x[0]?

original=Y
predicted=model.predict(X)


plt.rc('font', family='serif')
plt.figure()
plt.ylim([0, 2])
#plt.plot(Y, linestyle='dotted', color='#aaaaaa')
#plt.plot(original, linestyle='dashed', color='black')
plt.plot(predicted-Y, color='black')
plt.show()

val_loss=hist.history['val_loss']
val_mae=hist.history['val_mean_absolute_error']
tra_mae=hist.history['mean_absolute_error']

plt.rc('font',family='serif')
plt.figure()
plt.plot(range(len(val_mae)),val_mae,label='val_mae',color='black')
plt.plot(range(len(tra_mae)),tra_mae,label='tra_mae',color='b')
plt.xlabel('epochs')
plt.show()

print('hidden:20,batch:10,lr=0.001,b1,0.99,b2=0.9999,val_loss:0.0013')
print('hidden:10,batch:10,lr=0.001,b1,0.99,b2=0.9999,val_loss:0.00098')
print('hidden:10,batch:10,lr=0.001,b1,0.9,b2=0.9999,val_loss:0.00083')
