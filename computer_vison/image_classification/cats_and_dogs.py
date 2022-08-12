"""
@Time : 2021/6/1 10:47
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
import os
import keras
from keras_preprocessing import image as imageprocess
image_width,image_height = 150,150
train_dir = "./data/dogs_and_cats/train"
test_dir = "./data/dogs_and_cats/test"
no_classes = 2
no_validation = 50
epochs = 2
batch_size = 10
no_train = 200
no_test = 50
input_shape = (image_width,image_height,3)
epoch_steps = no_train // batch_size#也就是一个epoch会更新多少次参数，换句话说是一个batch更新一次
test_steps = no_test // batch_size

generator_train =imageprocess.ImageDataGenerator(rescale=1./255,horizontal_flip=True,zoom_range=0.3,shear_range=0.3)

generator_test = imageprocess.ImageDataGenerator(rescale=1./255,horizontal_flip=True,zoom_range=0.3,shear_range=0.3)
#flow_from_directory是根据文件夹来判断类别的。batch_size相当于生成器每次返回的数据量
train_images = generator_train.flow_from_directory(train_dir,batch_size=batch_size,target_size=(image_height,image_width))

test_images = generator_test.flow_from_directory(train_dir,batch_size=batch_size,target_size=(image_height,image_width))

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
simple_cnn = simple_cnn(input_shape)
#generator即生成器每次返回的数据量最好跟总数据量除以steps_per_epoch的大小一致，这样每个epoch正好使用一遍全部的训练数据
simple_cnn.fit_generator(train_images,steps_per_epoch=epoch_steps,epochs=epochs,validation_data=test_images,validation_steps=test_steps)
#fit和fit_generator的区别在于：fit和fit_generator的第一个参数都是训练数据，不同的是fit的该数据类型是numpy数组，而后者是一个生成器或则keras.utils.Sequence。前者是一下子把训练数据放入内存，batch_size参数意味着一次运算处理的数据量大小（该批数据将被送到CPU或GPU运算)。后者没有batch_size参数。而是steps_per_epoch，也就是生成器在一个epoch中返回多少次数据，而且合理的设置是生成器每次返回的数据量和前者的batch_size指定的大小一样，也就是和总数据量除以steps的大小应该一样。对于validation_data的原理也是这样。所以对于小数据量即占用内存不大的情况下可以用fit，更推荐用生成器模式
#用生成器喂数据的优点，除了不依赖大内存以外，还具有非常多的优点。比如，它可以并行预处理数据，咱们的模型是运行在GPU上的，generator运行在CPU上，所以GPU跑模型，CPU预处理数据，这样就可以达到很高的时效性。











