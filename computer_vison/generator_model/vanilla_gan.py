"""
@Time : 2021/6/2 17:41
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
import keras


def convolution_layer(input_layer,filters,kernel_size=[4,4],activation=tf.nn.leaky_relu):
    layer = tf.layers.conv2d(inputs=input_layer,filters=filters,kernel_size=kernel_size,activation=activation,kernel_regularizer=tf.nn.l2_loss,bias_regularizer=tf.nn.l2_loss)
    return layer

def pooling_layer(input_layer,
                  pool_size=[2, 2],
                  strides=2):
    layer = tf.layers.max_pooling2d(
        inputs=input_layer,
        pool_size=pool_size,
        strides=strides
    )
    return layer
#对转置卷积的需要一般来自希望使用与正常卷积相反方向的变换,即从具有某种卷积输出形状的某物到具有其输入形状的物体,同时保持与所述卷积兼容的连接模式
def transpose_convolution_layer(input_layer,filters,kernel_size=[4,4],activation=tf.nn.relu,strides=2):
    layer = tf.layers.conv2d_transpose(inputs=input_layer,filters=filters,kernel_size=kernel_size,activation=activation,strides=strides,kernel_regularizer=tf.nn.l2_loss,bias_regularizer=tf.nn.l2_loss)
    return layer

def dense_layer(input_layer,units,activation=tf.nn.relu):
    layer = tf.layers.dense(inputs=input_layer,units=units,activation=activation)
    return layer

"""
现在定义一个生成器，它将噪声作为输入并转换为图像。生成器由一对全连接层组成，然后是对噪声进行上采样的转置卷积层。
最后，卷积层将噪声作为单个通道。每层之间都有批归一化层，可以使梯度平滑流动
"""
def get_generator(input_noise,is_training=True):
    generator = dense_layer(input_noise,1024)
    generator = tf.layers.batch_normalization(generator,training=is_training)
    generator = dense_layer(generator,7*7*256)
    generator = tf.layers.batch_normalization(generator,training=is_training)
    generator = tf.reshape(generator,[-1,7,7,256])
    generator = transpose_convolution_layer(generator,5)#示例是64
    generator = tf.layers.batch_normalization(generator,training=is_training)
    generator = transpose_convolution_layer(generator,5)#示例是32
    generator = tf.layers.batch_normalization(generator,training=is_training)
    generator = convolution_layer(generator,3)
    generator = convolution_layer(generator,1,activation=tf.nn.tanh)
    return generator

def get_discriminator(image,is_training=True):
    x_input_shape = tf.reshape(image,[-1,28,28,1],name='input_reshape')
    discriminator = convolution_layer(x_input_shape,5)#示例是64
    discriminator = convolution_layer(discriminator,5)#示例是128
    discriminator = tf.layers.flatten(discriminator)
    discriminator = dense_layer(discriminator,1024)
    discriminator = tf.layers.batch_normalization(discriminator,training=is_training)
    discriminator = dense_layer(discriminator,2)
    return discriminator

batch_size = 32
input_dimension = [227,227]
real_images = None
#创建一个噪声向量作为生成器的输入
input_noise = tf.random_normal([batch_size,*input_dimension])

gan = tf.contrib.gan.gan_model(get_generator,get_discriminator,real_images,input_noise)

tf.contrib.gan.gan_train(tf.contrib.gan.gan_train_ops(gan,tf.contrib.gan.gan_loss(gan,generator_loss_fn=tf.contrib.gan.losses.wasserstein_generator_loss,discriminator_loss_fn=tf.contrib.gan.losses.wasserstein_discriminator_loss),tf.train.AdamOptimizer(0.001),tf.train.AdadeltaOptimizer(0.0001)))

