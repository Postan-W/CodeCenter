"""
@Time : 2021/5/27 15:24
@Author : wmingzhu
@Annotation : 
"""
"""
注意：Filters的数量越多，同一张图片经过这些filters提取特征后占用显卡内存数量越大，经过这两层众多数量的filter卷积后，占用的内存极大。还和batchsize有关，batch一批一批进入显卡训练，因此batch越大，占用内存越大。
"""
#这里设置域是为了获得更好的可视化效果
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.saved_model import signature_constants, signature_def_utils, tag_constants, utils

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

mnist_data = input_data.read_data_sets('MNIST_data',one_hot=True)

input_size = 784
no_classes = 10
batch_size = 100
total_batches = 200
x_input = tf.placeholder(tf.float32,shape=[None,input_size])
y_input = tf.placeholder(tf.float32,shape=[None,no_classes])



def add_variable_summary(tf_variable,summary_name):
    with tf.name_scope(summary_name+"_summary"):
        mean = tf.reduce_mean(tf_variable)
        tf.summary.scalar('Mean',mean)
        with tf.name_scope('standard_deviation'):
            standard_deviation = tf.sqrt(tf.reduce_mean(tf.square(tf_variable-mean)))

        tf.summary.scalar('StandardDeviation',standard_deviation)
        tf.summary.scalar('Maximum',tf.reduce_max(tf_variable))
        tf.summary.scalar('Minimum',tf.reduce_min(tf_variable))
        tf.summary.histogram('Histogram',tf_variable)

x_input_reshape = tf.reshape(x_input,[-1,28,28,1],name='input_reshape')#其中-1指的是批次大小可以是任意的

def convolution_layer(input_layer,filters,kernel_size=[3,3],activation=tf.nn.relu):
    layer = tf.layers.conv2d(inputs=input_layer,filters=filters,kernel_size=kernel_size,activation=activation)
    add_variable_summary(layer,'convolution')
    return layer

def pooling_layer(input_layer,pool_size=[2,2],strides=2):
    layer = tf.layers.max_pooling2d(inputs=input_layer,pool_size=pool_size,strides=strides)
    add_variable_summary(layer,'pooling')
    return layer

def dense_layer(input_layer,units,activation=tf.nn.relu):
    layer = tf.layers.dense(inputs=input_layer,units=units,activation=activation)
    add_variable_summary(layer,'dense')
    return layer

convolution_layer_1 = convolution_layer(x_input_reshape,5)#filters数量根据机器性能要改动
pooling_layer_1 = pooling_layer(convolution_layer_1)
convolution_layer_2 = convolution_layer(pooling_layer_1,6)
pooling_layer_2 = pooling_layer(convolution_layer_2)
flattened_pool = tf.reshape(pooling_layer_2,[-1,5*5*6],name='flattened_pool')#进入密集连接层前将数据展平.-1是指样本维是多少都行，或者说不考虑样本维。5*5是从最后池化层出来时图像的大小，6是最后的卷积核数量，可以理解为图像深度。实际上feed_dict中送入的是一个batch的数据，所以相当于把这么多数据展平
dense_layer_bottleneck = dense_layer(flattened_pool,1024)

dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(inputs=dense_layer_bottleneck,rate=0.4,training=dropout_bool)

#将dropout层送入最后一个dense层
logits = dense_layer(dropout_layer,no_classes)
result = tf.nn.softmax(logits)
with tf.name_scope("loss"):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,logits=logits)
    loss_operation = tf.reduce_mean(softmax_cross_entropy,name='loss')
    tf.summary.scalar('loss',loss_operation)
with tf.name_scope('optimiser'):
    optimizer = tf.train.AdamOptimizer().minimize(loss_operation)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        predictions = tf.argmax(logits,1)
        correct_predictions = tf.equal(predictions,tf.argmax(y_input,1))

    with tf.name_scope('accuracy'):
        accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))
        tf.summary.scalar('accuracy',accuracy_operation)

# session = tf.Session(config=config)
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    merge_summary_operation = tf.summary.merge_all()#summary必须合并,这样的话执行add_summary时将全部写入
    train_summary_writer = tf.summary.FileWriter('./summary1/train',session.graph)
    test_summary_writer = tf.summary.FileWriter('./summary1/test')

    test_images,test_labels = mnist_data.test.images,mnist_data.test.labels
    for batch_no in range(total_batches):
        mnist_batch = mnist_data.train.next_batch(batch_size)
        train_images,train_labels = mnist_batch[0],mnist_batch[1]
        _,merged_summary = session.run([optimizer,merge_summary_operation],feed_dict={x_input:train_images,y_input:train_labels,dropout_bool:True})
        train_summary_writer.add_summary(merged_summary,batch_no)
        if batch_no %10 == 0:
            print("第",batch_no,"的准确度是:",session.run(accuracy_operation,feed_dict={x_input:test_images,y_input:test_labels,dropout_bool:False}))
            merged_summary,_ = session.run([merge_summary_operation,accuracy_operation],feed_dict={x_input:test_images,y_input:test_labels,dropout_bool:False})
            test_summary_writer.add_summary(merged_summary,batch_no)














