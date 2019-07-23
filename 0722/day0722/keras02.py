#1. 데이터
import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
x2 = np.array([4, 5, 6])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential() # Sequential -> 순서대로 내려가는 모델을 만들겠다

model.add(Dense(5, input_dim = 1, activation = 'relu')) # input레이어 = 1개노드
model.add(Dense(3))
model.add(Dense(4))
model.add(Dense(1)) # 출력을 하나로 만듦 # output 레이어

#정확도를 올리기 위해 바꿀수 있는것(1. 레이어의 깊이(개수), 2. 노드의 개수(Dense), 3. 학습횟수)

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
# model.fit(x, y, epochs = 200, batch_size = 11)
# epochs번 학습 / batch_size => 학습할때 한번에 할 단위(잘라서 실행할 단위)
# fit => x와 y를 batch_size씩 epochs번 학습하겠다.
model.fit(x, y, epochs = 100)

#4. 평가예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print('acc : ', acc)

y_predict = model.predict(x2) # x2라는 값을 넣을테니 예측을 해달라
print(y_predict)
