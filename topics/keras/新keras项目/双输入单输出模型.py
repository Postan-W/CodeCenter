import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_labels = len(np.unique(y_train))
y_train = to_categorical(y_train)
print("前2个标签:{}".format(y_train[:2]))
y_test = to_categorical(y_test)
image_size = x_train.shape[1]
x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

input_shape = (image_size, image_size, 1)
batch_size = 32
kernel_size = 3
dropout = 0.4
n_filters = 32
left_inputs = Input(shape=input_shape)
x = left_inputs
filters = n_filters
for i in range(3):
    x = Conv2D(filters=filters, kernel_size=kernel_size,padding='same',activation='relu')(x)
    x = Dropout(dropout)(x)
    x = MaxPooling2D()(x)
    filters *= 2

right_inputs = Input(shape=input_shape)
y = right_inputs
filters = n_filters
for i in range(3):
    y = Conv2D(filters=filters,
               kernel_size=kernel_size,
               padding='same',
               activation='relu',
               dilation_rate=2)(y)
    y = Dropout(dropout)(y)
    y = MaxPooling2D()(y)
    filters *= 2

y = concatenate([x, y])
#这里显示一下concatenate的效果
testdata1 = np.array([[[1,1,1,1],[2,2,2,2],[3,3,3,3]],[[4,4,4,4],[5,5,5,5],[6,6,6,6]]])
testdata2 = np.array([[[7,7,7,7],[8,8,8,8],[9,9,9,9]],[[10,10,10,10],[11,11,11,11],[12,12,12,12]]])
con_result = concatenate([testdata1,testdata2])
print(con_result)#可知拼接的是最里层的数据
y = Flatten()(y)
y = Dropout(dropout)(y)
outputs = Dense(num_labels, activation='softmax')(y)
model = Model([left_inputs, right_inputs], outputs)

def start():
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit([x_train, x_train],
              y_train,
              validation_data=([x_test, x_test], y_test),
              epochs=20,
              batch_size=batch_size)

    score = model.evaluate([x_test, x_test],
                           y_test,
                           batch_size=batch_size,
                           verbose=0)
    print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
