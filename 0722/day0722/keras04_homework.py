#1. 데이터
import numpy as np
# 훈련시킬 데이터
x_train = np.array([i for i in range(1, 101)])
y_train = np.array([i-400 for i in range(501, 601)])
# 테스트할 데이터
x_test = np.array([i for i in range(1001, 1101)])
y_test = np.array([i for i in range(1101, 1201)])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() # Sequential -> 순서대로 내려가는 모델을 만들겠다

model.add(Dense(5, input_dim = 1, activation = 'relu'))
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1)) # 출력을 하나로 만듦 # output 레이어

# model.summary() # param #에 나오는 값 = (입력값 * 그 전 레이어의 값) + 바이어스(입력값과 동일)
# param(line)의 개수로 생각해서 그림 그려서 그림개수를 따져본다. [line수 + 바이어스]
# y = wx+b => wx = line / b = 바이어스 / y = param(파라메터값)

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 50, batch_size = 1)
# model.fit(x_train, y_train, epochs = 300)

#4. 평가예측
loss, acc = model.evaluate(x_test, y_test, batch_size = 1)
print('acc : ', acc)

y_predict = model.predict(x_test)
print(y_predict)
