#1. 데이터
import numpy as np

# x = np.array(range(1, 101)) # x[0] => 1임
# y = np.array(range(1, 101)) # y[0] => 1임
x = np.array([range(100), range(311, 411), range(100)]) # 2행 100열이됨
y = np.array([range(501, 601), range(711, 811), range(100)])

print(x.shape)
print(y.shape)

x = np.transpose(x) # 100행 2열로 바꿔줌
y = np.transpose(y)

print(x.shape)
print(y.shape)

# 싸이킷런의 트래인 테스트 스플릿을 가져옴
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state = 66, test_size = 0.5)

print(x_test.shape)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential() # Sequential -> 순서대로 내려가는 모델을 만들겠다

input1 = Input(shape=(3, ))
dense1 = Dense(10, activation = 'relu')(input1)
dense2 = Dense(20)(dense1)
dense3 = Dense(10)(dense2)
dense4 = Dense(10)(dense3)
dense5 = Dense(10)(dense4)
denseE = Dense(3)(dense5)

model = Model(input = input1, output = denseE)
model.summary()

# model.add(Dense(10, input_shape = (3, ), activation = 'relu')) # (1, ) => 몰라행 1열(행이 무시가 되고 열로만 판단)
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(10))
# model.add(Dense(3)) # 2열이 되면서 시작인풋도 2로 바뀌고 shape도(2, )로 바뀌고 최종아웃풋도 2로바뀜

#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
# model.fit(x_train, y_train, epochs = 100)
# model.fit(x_train, y_train, epochs = 100, batch_size = 1)
model.fit(x_train, y_train, epochs = 500, batch_size = 2, validation_data=(x_val, y_val))

#4. 평가,예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)
print('acc : ', acc)

y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
print("RMSE :", RMSE(y_test, y_predict))

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)
