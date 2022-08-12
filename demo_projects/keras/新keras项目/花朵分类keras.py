from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from 图片处理 import get_images_labels
images,labels = get_images_labels()
sep = int(len(images)*0.9)
x_train,y_train = images[:sep],labels[:sep]
x_test,y_test = images[sep:],labels[sep:]

input_shape = (x_train.shape)[1:]
batch_size = int(len(x_train)/10)
kernel_size = 3
filters = 64
dropout = 0.3
classes = 5

inputs = Input(shape=input_shape,name="inputtest")
# y = Conv2D(filters=filters,
#            kernel_size=kernel_size,
#            activation='tanh')(inputs)
# y = MaxPooling2D()(y)
# y = Conv2D(filters=filters,
#            kernel_size=kernel_size,
#            activation='tanh')(y)
# y = Conv2D(filters=filters,
#            kernel_size=kernel_size,
#            activation='relu')(y)
y = Flatten()(inputs)
y = Dense(2,activation="tanh")(y)
y = Dropout(dropout)(y)
outputs = Dense(5, activation='softmax',name="outputtest")(y)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
model.fit(x_train,y_train,validation_data=(x_test, y_test),epochs=10,batch_size=batch_size)
score = model.evaluate(x_test,
                       y_test,
                       batch_size=batch_size,
                       verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * score[1]))
model.save("./models/flowers_cli_dense.h5")
