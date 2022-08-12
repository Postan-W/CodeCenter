"""
@Time : 2021/5/28 15:43
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
batch_size = 128
no_classes = 10
epochs = 2
image_height,image_width = 28,28

(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
#将向量重整为图片形式
x_train = x_train.reshape(x_train.shape[0],image_height,image_width,1).astype('float32')/255
x_test = x_test.reshape(x_test.shape[0],image_height,image_width,1).astype('float32')/255
input_shape = (image_height,image_width,1)

y_train = tf.keras.utils.to_categorical(y_train,no_classes)
y_test = tf.keras.utils.to_categorical(y_test,no_classes)

def simple_cnn(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=5,kernel_size=(3,3),activation='relu',input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(filters=5,kernel_size=(3,3),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024,activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes,activation='softmax'))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    return model

simple_cnn_model = simple_cnn(input_shape)

simple_cnn_model.fit(x_train,y_train,batch_size,epochs,(x_test,y_test))

test_loss,test_accuracy = simple_cnn_model.evaluate(x_test,y_test,verbose=0)
print("测试损失:",test_loss,"测试准确度:",test_accuracy)
# simple_cnn_model.save("simple_cn_model.h5")















