#1. 데이터
import numpy as np

x1 = np.array([range(100), range(311, 411), range(100)])
y1 = np.array([range(501, 601), range(711, 811), range(100)])
x2 = np.array([range(100, 200), range(311, 411), range(100, 200)])
y2 = np.array([range(501, 601), range(711, 811), range(100)])

x1 = np.transpose(x1)
y1 = np.transpose(y1)
x2 = np.transpose(x2)
y2 = np.transpose(y2)

print(x1.shape)
print(y1.shape)
print(x2.shape)
print(y2.shape)

# 싸이킷런의 트래인 테스트 스플릿을 가져옴
from sklearn.model_selection import train_test_split
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state = 66, test_size = 0.4)
x1_test, x1_val, y1_test, y1_val = train_test_split(x1_test, y1_test, random_state = 66, test_size = 0.5)

from sklearn.model_selection import train_test_split
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state = 66, test_size = 0.4)
x2_test, x2_val, y2_test, y2_val = train_test_split(x2_test, y2_test, random_state = 66, test_size = 0.5)

print(x1_train.shape)
print(x1_test.shape)
print(x1_val.shape)
print(y1_train.shape)
print(y1_test.shape)
print(y1_val.shape)

print(x2_train.shape)
print(x2_test.shape)
print(x2_val.shape)
print(y2_train.shape)
print(y2_test.shape)
print(y2_val.shape)

#2. 모델구성
from keras.models import Sequential, Model
from keras.layers import Dense, Input
# model = Sequential() # Sequential -> 순서대로 내려가는 모델을 만들겠다

input1 = Input(shape=(3, ))
dense1 = Dense(100, activation = 'relu')(input1)
dense1_2 = Dense(30)(dense1)
dense1_3 = Dense(7)(dense1_2)

input2 = Input(shape=(3, ))
dense2 = Dense(50, activation = 'relu')(input2)
dense2_2 = Dense(7)(dense2)

from keras.layers.merge import concatenate
merge1 = concatenate([dense1_3, dense2_2])

middle1 = Dense(10)(merge1)
middle2 = Dense(5)(middle1)
middle3 = Dense(7)(middle2)

################################ 요기부터 아웃풋 모델

output1 = Dense(30)(middle3)
output1_2 = Dense(7)(output1)
output1_3 = Dense(3)(output1_2)

output2 = Dense(20)(middle3)
output2_2 = Dense(7)(output2)
output2_3 = Dense(3)(output2_2)

model = Model(input = [input1, input2], output = [output1_3, output2_3])

model.summary()

#3. 훈련
# model.compile(loss = 'mse', optimizer = 'adam', metrics = ['accuracy'])
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['acc'])
# model.fit(x_train, y_train, epochs = 100)
# model.fit(x_train, y_train, epochs = 100, batch_size = 1)
model.fit([x1_train, x2_train], [y1_train, y2_train],
            epochs = 100, batch_size = 1, validation_data=([x1_val, x2_val], [y1_val, y2_val]))

#4. 평가,예측
_,_,_,acc1, acc2 = model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size = 1)
#언더바 => 값을 아예 입력하지않고 없게만듦
print('acc1 : ', acc1)
print('acc2 : ', acc2)

y_predict, y_predict2 = model.predict([x1_test, x2_test])
print(y_predict)
print(y_predict2)

#RMSE 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
#따로구해서 따로 출력
print("RMSE1 :", RMSE(y1_test, y_predict))
print("RMSE2 :", RMSE(y2_test, y_predict2))
#따로구한걸 더해서 2로 나눈것을 출력
print("RMSE :", (RMSE(y1_test, y_predict)+RMSE(y2_test, y_predict2))/2)

#R2 구하기
from sklearn.metrics import r2_score
#따로 구해서 따로 출력
r2_y_predict = r2_score(y1_test, y_predict)
r2_y_predict2 = r2_score(y2_test, y_predict2)
print("R2_1 :", r2_y_predict)
print("R2_2 :", r2_y_predict2)
#따로구한걸 더해서 2로 나눈것을 출력
print("R2 :", (r2_y_predict + r2_y_predict2)/2)
