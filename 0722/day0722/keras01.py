#1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. 모델구성
from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
# input레이어로 1개로 시작 한 후 히든레이어로 이어진다음 아웃풋레이어로 끝
                                                        # dim = 디멘션 = 차원
model.add(Dense(5, input_dim = 1, activation = 'relu')) # 레이어생성 5개의 노드 생성해 진행 # input레이어 = 1개노드
model.add(Dense(3)) # 2번째 레이어 생성 3개의 노드 생성해 진행 # hidden레이어 - 컴퓨터가 스스로 훈련하고 학습하는 부분임
model.add(Dense(4)) # 3번째 레이어 생성 4개의 노드 생성해 진행 # hidden레이어 - 그래서 정확히 알 수가 없기에 히든레이어
model.add(Dense(1)) # 출력을 하나로 만듦 # output 레이어

#3. 훈련
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x, y, epochs = 100, batch_size = 1) # 100번 학습

#4. 평가예측
loss, acc = model.evaluate(x, y, batch_size = 1)
print('acc : ', acc)
