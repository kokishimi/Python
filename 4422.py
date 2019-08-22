# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 14:21:04 2019

@author: kokis
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,BatchNormalization
from keras.optimizers import SGD, Adadelta,RMSprop,Adam
from keras.callbacks import EarlyStopping
#from keras.initializers import TruncatedNormal
import matplotlib.pyplot as plt


np.random.seed()

mnist=datasets.fetch_mldata('MNIST original')

n=len(mnist.data)
N=30000
N_train=20000
N_validation=4000
indices=np.random.permutation(range(n))[:N]
X=mnist.data[indices]
#Standarize
#X=X/X.max()
#X=X-X.mean(axis=1).reshape(len(X),1)
y=mnist.target[indices]
Y=np.eye(10)[y.astype(int)]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=N_train)
X_train,X_validation,Y_train,Y_validation=\
    train_test_split(X_train,Y_train,test_size=N_validation)

n_in=len(X[0])
n_hiddens=[200,200,200,200,200]
n_out=len(Y[0])
activation='relu'
p_keep=0.2

model=Sequential()
for i, input_dim in enumerate(([n_in]+n_hiddens)[:-1]):
    model.add(Dense(n_hiddens[i],input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(p_keep))

model.add(Dense(n_out))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adadelta(rho=0.95),
              metrics=['accuracy'])

epochs=50
batch_size=100

early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)

hist=model.fit(X_train,Y_train,epochs=epochs,
               batch_size=batch_size,
               validation_data=(X_validation,Y_validation),
               callbacks=[early_stopping])

loss_and_metrics=model.evaluate(X_test,Y_test)

val_loss=hist.history['val_loss']
val_acc=hist.history['val_acc']
tra_acc=hist.history['acc']

plt.rc('font',family='serif')
fig=plt.figure()
plt.plot(range(len(val_acc)),val_acc,label='val_acc',color='black')
plt.plot(range(len(tra_acc)),tra_acc,label='tra_acc',color='b')
plt.xlabel('epochs')
plt.show()
# plt.savefig('mnist_keras.eps')

print(loss_and_metrics)


print('sig 1 400: 93%')
print('sig 1 800: 92%')
print('tah 3 200: 93%')
print('tah 4 200: 93%')
print('tah 1 200: 92%')
print('tah 4 200: 92%')
print('rel 5 400: 93% epo100 lr0.001 do0.2')
print('testdatamiss')
print('testvalidation,N=30000: 95%')
print('testvalidation,N=30000,momenum0.5: 97%')
print('testvalidation,N=30000,momenum0.5nestrov: 97%')
print('tvt,epo30,N=30000,adadelta0.95: 99%')
print('tvt,epo30,N=30000,RMSprop0.0001: 98%')
print('tvt,epo30,N=30000,Adam0.001b10.9b20.999: 98%')
print('tvt,epo50,N=30000,Adam0.0001b10.9b20.999,ES: 98%')
print('tvt,epo50,N=30000,Adam0.0001b10.9b20.999,ES,BN: 99.5%')
print('tvt,epo50,N=30000,Adam0.001b10.9b20.999,ES,BN: 99.5%')
print('tvt,epo50,N=30000,Adadelta0.95,ES,BN: 99.6%')
print('actual')
print('tvt,epo50,N=30000,Adadelta0.95,ES,BN: 97%')
list=model.get_weights()
print(list[0])