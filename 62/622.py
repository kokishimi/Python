# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:24:44 2019

@author: kokis
"""

import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import RepeatVector
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn import datasets
    
np.random.seed()

def n(digits=3):
    number=''
    for i in range(np.random.randint(1,digits+1)):
        number+=np.random.choice(list('0123456789'))
        
    return int(number)

def padding(chars,maxlen):
    return chars+' '*(maxlen-len(chars))

digits=3
input_digits=digits*2+1
output_digits=digits+1

added=set()
questions=[]
answers=[]

N=20000
N_train=16000
N_validation=3200

while len(questions)<N:
    a,b=n(),n()
    
    pair=tuple(sorted((a,b)))
    if pair in added:
        continue
    
    question='{}+{}'.format(a,b)
    question=padding(question,input_digits)
    answer=str(a+b)
    answer=padding(answer,output_digits)
    
    added.add(pair)
    questions.append(question)
    answers.append(answer)
    
chars='0123456789+ '
char_indices=dict((c,i) for i, c in enumerate(chars))
indices_char=dict((i,c) for i, c in enumerate(chars))

X=np.zeros((len(questions),input_digits,len(chars)),dtype=np.integer)
Y=np.zeros((len(questions),output_digits,len(chars)),dtype=np.integer)

for i in range(N):
    for t, char in enumerate(questions[i]):
        X[i,t,char_indices[char]]=1
        
    for t, char in enumerate(answers[i]):
        Y[i,t,char_indices[char]]=1

X_train,X_test,Y_train,Y_test=\
    train_test_split(X,Y,train_size=N_train)

X_train,X_validation,Y_train,Y_validation=\
    train_test_split(X_train,Y_train,test_size=N_validation)
    
n_in=12
n_hidden=128
n_out=12

model=Sequential()

#encoder
model.add(LSTM(n_hidden,input_shape=(input_digits,n_in)))

#decoder
model.add(RepeatVector(output_digits))
model.add(LSTM(n_hidden,return_sequences=True))

model.add(TimeDistributed(Dense(n_out)))
model.add(Activation('softmax'))

optimizer=Adam(lr=0.01,beta_1=0.9,beta_2=0.9999)
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