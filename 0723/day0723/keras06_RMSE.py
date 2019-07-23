#1. 데이터
import numpy as np
# 훈련시킬 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 테스트할 데이터
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x3 = np.array([101, 102, 103, 104, 105, 106])
x4 = np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65])
x5 = np.array(range(30, 50)) # 30에서 49까지 넣음(19개)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() # Sequential -> 순서대로 내려가는 모델을 만들겠다

# model.add(Dense(10, input_dim = 1, activation = 'relu'))
model.add(Dense(10, input_shape = (1, ), activation = 'relu')) # (1, ) => 몰라행 1열(행이 무시가 되고 열로만 판단)
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1)) # 출력을 하나로 만듦 # output 레이어

model.summary() # param #에 나오는 값 = (입력값 * 그 전 레이어의 값) + 바이어스(입력값과 동일)
# param(line)의 개수로 생각해서 그림 그려서 그림개수를 따져본다. [line수 + 바이어스]
# y = wx+b => wx = line / b = 바이어스 / y = param(파라메터값)

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 500, batch_size = 2)
# model.fit(x_train, y_train, epochs = 500)

#4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)
print('acc : ', acc)

y_predict = model.predict(x_test)
print(y_predict)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

print("RMSE :", RMSE(y_test, y_predict))
