#1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# x2 = np.array([4, 5, 6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() # Sequential -> 순서대로 내려가는 모델을 만들겠다

model.add(Dense(5, input_dim = 1, activation = 'relu')) # input레이어 = 1개노드
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1)) # 출력을 하나로 만듦 # output 레이어

model.summary() # param #에 나오는 값 = (입력값 * 그 전 레이어의 값) + 바이어스(입력값과 동일)
# param(line)의 개수로 생각해서 그림 그려서 그림개수를 따져본다. [line수 + 바이어스]
# y = wx+b => wx = line / b = 바이어스 / y = param(파라메터값)
# 바이어스는 전지전능하여 모든곳에 다있다. 기존 노드수에 하나를 추가하는 느낌으로 계산
# 2*5 => 10 / 3*6 => 18 ...
# 머신을 할때 바이어스는 언제나 모든곳에 붙어서 진행된다(파라메터의 숫자를 구할때)

#정확도를 올리기 위해 바꿀수 있는것(1. 레이어의 깊이(개수), 2. 노드의 개수(Dense), 3. 학습횟수)

# #3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
# # model.fit(x, y, epochs = 200, batch_size = 11)
# # epochs번 학습 / batch_size => 학습할때 한번에 할 단위(잘라서 실행할 단위)
# # fit => x와 y를 batch_size씩 epochs번 학습하겠다.
# model.fit(x, y, epochs = 100)

# #4. 평가예측
# loss, acc = model.evaluate(x, y, batch_size = 1)
# print('acc : ', acc)

# y_predict = model.predict(x2) # x2라는 값을 넣을테니 예측을 해달라
# print(y_predict)
