"""
@Time : 2021/6/1 16:41
@Author : wmingzhu
@Annotation : 
"""
#迁移学习是从一个大型数据集上预训练过的模型中学习的过程

import tensorflow as tf
import keras
import numpy as np
train_dir = "./data/dogs_and_cats/train"#两类各100共200张图片
test_dir = "./data/dogs_and_cats/test"
epoch_steps = 200 / 10 #训练图片个数除以batch_size
test_steps = 50 / 10

# print(train_labels)
# print("VGGNET16模型结构:",model.summary())
def build():
    generator = keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)  # 数据增强方法
    # 内置模型放在用户目录.keras/models下,可以下载后放在这里
    model = keras.applications.vgg16.VGG16(include_top=False)  # 该参数表示不保留末尾的dense层
    train_images = generator.flow_from_directory(train_dir, batch_size=10, target_size=(150, 150), class_mode=None,
                                                 shuffle=False)

    train_bottleneck_features = model.predict_generator(train_images, epoch_steps)  # 就前向传递得到一个结果，作为后面的新加入层的结果

    test_images = generator.flow_from_directory(test_dir, batch_size=10, target_size=(150, 150), class_mode=None,
                                                shuffle=False)
    test_bottleneck_features = model.predict_generator(test_images, test_steps)
    print(train_bottleneck_features.shape)
    train_labels = np.array([0] * 100 + [1] * 100)
    test_labels = np.array([0] * 25 + [1] * 25)
    # 用的是多分类交叉熵损失，所以这里标签要one-hot化
    train_labels = keras.utils.to_categorical(train_labels)
    test_labels = keras.utils.to_categorical(test_labels)
    model = keras.models.Sequential()
    print(type(model))
    model.add(keras.layers.Flatten(input_shape=train_bottleneck_features.shape[1:]))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(2, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print("自定义层结构:",model.summary())
    model.fit(train_bottleneck_features, train_labels, batch_size=10, epochs=2,
              validation_data=(test_bottleneck_features, test_labels))

#上面是使用加载的VGG16预测出的结果送入自定义层。下面是直接将VGG16作为底层，然后固化前25层
from cats_and_dogs import train_images,test_images
def build2():
    #如果你的 weight 不设置为 None，那么系统会自动从网上给你下载 vgg-16的参数，巨巨巨巨大。所以建议 weight=None
    #请注意，input_shape 和 pooling参数只能在 include_top 是 False 时指定。因为如果默认使用完整的 vgg-16网络，这个时候全连接层的参数是固定的，如果随便调了输入图片的尺寸或者 pooling 的参数，那么在最终特征图 feature map 到达 dense 层的时候，就不一定满足神经元个数匹配，因此只有在不包含全连接层（include_top=False）的时候才能指定修改 input_shape 以及 pooling 参数
    #注意，在实例化网络的时候要以上面的方式添加参数，如果你按照下面的方式创建网络，会报错：AttributeError: ‘NoneType’ object has no attribute 'image_data_format’
    vggmodel = keras.applications.vgg16.VGG16(input_shape=[150, 150, 3], include_top=False, backend=keras.backend,
                           layers=keras.layers,weights=None,
                           models=keras.models,
                           utils=keras.utils)
    print("vgg输出形状：",vggmodel.output_shape)
    print(len(vggmodel.layers))
    for vgg_layer in vggmodel.layers[:25]:
        vgg_layer.trainable = False
    model_top = keras.models.Sequential()
    model_top.add(vggmodel)
    model_top.add(keras.layers.Flatten(input_shape=vggmodel.output_shape[1:]))
    model_top.add(keras.layers.Dense(256, activation='relu'))
    model_top.add(keras.layers.Dropout(0.5))
    model_top.add(keras.layers.Dense(2, activation='softmax'))
    model_top.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    print(model_top.summary())
    #Kerasfit函数接受数据数组，numpy数组，而不是生成器。您需要的函数是fit_generator。注意，fit_generator采用稍有不同的参数，例如steps_per_epoch，而不是batch_size。
    # model_top.fit(image_for_predict, train_labels, batch_size=10, epochs=2,
    #           validation_data=(test_bottleneck_features, test_labels))
    model_top.fit_generator(train_images,epochs=2,steps_per_epoch=20,validation_data=test_images,validation_steps=5)

build2()
# build()
