# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:21:46 2019

@author: kokis
"""

import numpy as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,BatchNormalization
from keras.optimizers import Adadelta,SGD,Adam
from keras.callbacks import EarlyStopping
#from keras.initializers import TruncatedNormal
from keras import backend as K

import matplotlib.pyplot as plt

#import csv
import pandas as pd

def R2eval(y_true,y_pred):
    return K.abs(K.var(y_pred)/K.var(y_true))

XYpd = pd.read_csv('2018456ac30IDCCDummyforNN.csv')
Xpd= XYpd.drop('log10Dist', axis=1)

print('Ximport')

Ypd = pd.read_csv('2018456ac30IDCCDummyforNN.csv',usecols=['log10Dist'])

print('import')

np.random.seed()

X = Xpd.values
Y = Ypd.values

X_train,X_test,y_train,y_test=train_test_split(X,Y,train_size=0.8)

print('split')
#Standarize
#X=X/X.max()
#X=X-X.mean(axis=1).reshape(len(X),1)
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

X_train_mean=X_train.mean(axis=0)
X_train_std=X_train.std(axis=0)
X_train-=X_train_mean
X_train/=X_train_std

print('XNorm')

y_train_mean=y_train.mean()
y_train_std=y_train.std()
y_train-=y_train_mean
y_train/=y_train_std

print('YNorm')

X_test-=X_train_mean
X_test/=X_train_std
y_test-=y_train_mean
y_test/=y_train_std

print('testNorm')

n_in=len(X_train[0])
n_hiddens=[256]*5
activation='relu'
#activation='sigmoid'
p_keep=0.1

model=Sequential()
for i, input_dim in enumerate(([n_in]+n_hiddens)[:-1]):
    model.add(Dense(n_hiddens[i],input_dim=input_dim))
    model.add(BatchNormalization())
    model.add(Activation(activation))
    model.add(Dropout(p_keep))

model.add(Dense(1))
#model.add(Activation('sigmoid'))

model.compile(loss='mse',
              optimizer=Adadelta(rho=0.85),
              metrics=['mae',R2eval])

epochs=100
batch_size=1024

early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=1)

hist=model.fit(X_train,y_train,epochs=epochs,
               batch_size=batch_size,
               validation_split=0.2,
               callbacks=[early_stopping])

loss_and_metrics=np.array(model.evaluate(X_test,y_test))
y_pred = model.predict(X_test)
y_pred2=[]
for i in range(len(y_pred)):
    y_pred2.append(y_pred[i][0])
    
#mean should be 0
#child=np.sum(np.square(y_test-y_pred2))
child=np.sum(np.square(y_test-y_pred))
mother2=np.sum(np.square(y_test))

R24=1-(child/mother2)

#stdev=np.sqrt(np.sum(np.square(y_pred2-y_test))/len(y_pred))
stdev=np.sqrt(np.sum(np.square(y_pred-y_test))/len(y_pred))
msetest=np.mean(np.square(y_pred-y_test))


val_loss=hist.history['val_loss']
val_mae=hist.history['val_mean_absolute_error']
tra_mae=hist.history['mean_absolute_error']

plt.rc('font',family='serif')
fig=plt.figure()
plt.plot(range(len(val_mae)),val_mae,label='val_mae',color='black')
plt.plot(range(len(tra_mae)),tra_mae,label='tra_mae',color='b')
plt.xlabel('epochs')
plt.show()
# plt.savefig('mnist_keras.eps')

print(loss_and_metrics)
print('R24(better):',R24)
print('stdev:',stdev)
print('mse:',msetest)

print('layer4,rho0.85,batch64,DO0.1:R2=0.58,stdev=0.66')

list=model.get_weights()
print(list[0][0])

#with open('LAlnDfewV_train.csv', 'w') as f:
 #   writer = csv.writer(f)
  #  #y_tra2 = y_train.reshape(-1,1)
   # cwrite=np.hstack((X_train,y_train))
    #writer.writerows(cwrite)
    
#with open('LALnDfewV_test.csv', 'w') as f2:
 #   writer = csv.writer(f2)
  #  #y_tes2 = y_test.reshape(-1,1)
   # cwrite2=np.hstack((X_test,y_test))
    #writer.writerows(cwrite2)