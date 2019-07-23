#1. 데이터
import numpy as np

x = np.array(range(1, 101)) # x[0] => 1임
y = np.array(range(1, 101)) # y[0] => 1임

# x_train = x[0:60] # x[0](첫번째) 값인 1부터 x[59](60번째)60까지 x_frame에 넣는다
# y_train = y[0:60]

# x_val = x[60:80]
# y_val = y[60:80]

# x_test = x[80:100]
# y_test = y[80:100]

# 싸이킷런의 트래인 테스트 스플릿을 가져옴
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 66, test_size = 0.4)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state = 66, test_size = 0.5)

print(len(x_train),x_train)
print(len(y_train),y_train)
print(len(x_test),x_test)
print(len(y_test),y_test)
print(len(x_val),x_val)
print(len(y_val),y_val)

'''
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

model.summary()

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
'''