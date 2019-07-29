#-*- coding: utf-8 -*-

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import numpy
import os
import tensorflow as tf

#데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 # (60000, 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 # (10000, 28, 28, 1)
print(Y_train.shape) #(60000, )
print(Y_test.shape) #(10000, )
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print(Y_train.shape) #(60000, 10)
print(Y_test.shape) #(10000, 10)

# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)


#컨볼루션 신경망의 설정
model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPool2D(pool_size = 2)) # (2, 2)와 똑같음
model.add(Dropout(0.25)) # 25%를 잘라버린다(필요없는거..!)
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax')) # (이진분류 제외) 분류모델에서 마지막 아웃풋에 대한 activation은 무조건 softmax를 사용해야함

model.compile(loss = 'categorical_crossentropy', # 분류모델에서는 loss를 categorical_crossentropy를 쓴다
              optimizer = 'adam',
              metrics = ['accuracy'])

#모델 최적화 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss', patience = 10)

#모델의 설정
history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test),
                    epochs = 30, batch_size = 200, verbose = 1, 
                    callbacks = [early_stopping_callback])

# model.summary() # DNN과 파라메터 개수차이를 확인하기 위해 사용한 서머리함수

#테스트 정확도 출력
print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1])) #분류모델은 Accuracy가 정확
