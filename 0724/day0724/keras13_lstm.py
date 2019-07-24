from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

#1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7], [6,7,8],[7,8,9],[8,9,10],[9,10,11],[10,11,12]]) #10행3열로
y = array([4,5,6,7,8,9,10,11,12,13]) #10열로

print("x.shape :", x.shape) #(4, 3)
print("y.shape :", y.shape) #(4, ) => (1, 4)

x = x.reshape((x.shape[0], x.shape[1], 1))

print("x.shape :", x.shape) #(4, 3, 1) (행, 열, 몇개씩 자를 것인가) => 행은 무시되므로 돌아감

#2. 모델 구성
model = Sequential()
model.add(LSTM(8, activation = 'relu', input_shape = (3, 1)))
model.add(Dense(10))
model.add(Dense(9))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(6))
model.add(Dense(3))
model.add(Dense(1))

# model.summary()

#3. 실행
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x, y, epochs = 200)

x_input = array([11,12,13]) #(3, ) => (1, 3) =>몇개씩자를것인지가 없음(1, 3, ????)
x_input = x_input.reshape((1,3,1)) # => 1행 3열에 1개씩 자르는 것으로 바꿈 => 3열 1개씩!!!!!! 헷갈리지말기

yhat = model.predict(x_input)
print(yhat)

# 노드, 레이어, 에포s를 건드려서 결과값을 14로 만들어라
