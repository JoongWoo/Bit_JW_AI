from keras.models import Sequential

filter_size = 32
kernel_size = (3, 3)

from keras.layers import Conv2D, MaxPooling2D, Flatten
model = Sequential()
model.add(Conv2D(7, (2, 2), padding = 'same', input_shape = (10, 10, 1))) # => 흑백 / 칼라 => 3
# model.add(Conv2D(7, (2, 2), padding = 'valid', input_shape = (5, 5, 1)))
# padding => 채울 테두리의 크기를 지정하는 것, 옆에 잘리는 것이 없어짐
# model.add(Conv2D(16, (2, 2)))
model.add(MaxPooling2D(3, 3)) # 2,2만큼 잘라서 큰거끼리 모아! 이런느낌인듯 shape가 반으로 줄어든다 => 결과값보면 이해갈거임
#넘치는 것은 버립니다!!
# model.add(Conv2D(8, (2, 2)))
# model.add(Flatten()) # => 2D를 1D로 펴준다!

model.summary()
