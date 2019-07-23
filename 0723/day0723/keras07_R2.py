#1. 데이터
import numpy as np
# 훈련시킬 데이터
x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 테스트할 데이터
x_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y_test = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
x3 = np.array([101, 102, 103, 104, 105, 106])
x4 = np.array(range(30, 50)) # 30에서 49까지 넣음(19개)

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() # Sequential -> 순서대로 내려가는 모델을 만들겠다

# model.add(Dense(10, input_dim = 1, activation = 'relu'))
model.add(Dense(120, input_shape = (1, ), activation = 'relu')) # (1, ) => 몰라행 1열(행이 무시가 되고 열로만 판단)
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(140))
model.add(Dense(160))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(220))
model.add(Dense(240))
model.add(Dense(260))
model.add(Dense(280))
model.add(Dense(300))
model.add(Dense(360))
model.add(Dense(360))
model.add(Dense(300))
model.add(Dense(280))
model.add(Dense(260))
model.add(Dense(240))
model.add(Dense(220))
model.add(Dense(168))
model.add(Dense(1)) # 출력을 하나로 만듦 # output 레이어

model.summary()

#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mse'])
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

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

#R2 구하기
from sklearn.metrics import r2_score
r2_y_predict = r2_score(y_test, y_predict)
print("R2 :", r2_y_predict)
