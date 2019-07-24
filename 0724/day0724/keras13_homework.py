from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM
#70, 80, 90이 어떻게 나오게하는가?
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

model.summary()

#3. 실행
model.compile(optimizer = 'adam', loss = 'mse')
model.fit(x, y, epochs = 200)

x_input = array([11,12,13])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)

 # parameters = 4(nm+n2+n)