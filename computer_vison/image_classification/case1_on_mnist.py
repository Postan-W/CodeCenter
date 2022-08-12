"""
@Time : 2021/5/27 10:56
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist_data = input_data.read_data_sets('MNIST_data',one_hot=True)#下载地址:http://yann.lecun.com/exdb/mnist/，下载页面上的4个压缩包，然后放到MNIST_data文件夹下
input_size = 784
no_classes = 10
batch_size = 50
total_batches = 400
from tensorflow.saved_model import signature_constants, signature_def_utils, tag_constants, utils
def signature_defination(sess,saved_dir,x,pre):
    #存储了整个graph的信息，并且由于定义了signature，所以又指定了如输入输出这样的信息
    builder = tf.saved_model.builder.SavedModelBuilder(saved_dir)
    signature = signature_def_utils.predict_signature_def(inputs={"input_x":x},
                                                          outputs={"outputs":pre})
    #注：通过加载的代码可以看到，上面的inputs和outputs将作为meta_graph.signature_def["graph_information"]返回的signature对象的属性
    #tags的作用是为了区分图，通常这个名称 tag 以该计算图的功能和使用到的设备命名，比如 serving or training， CPU or GPU。一个图add一次
    builder.add_meta_graph_and_variables(sess=sess,tags=[tag_constants.SERVING],signature_def_map={'test_signature':signature})
    builder.save()
x_input = tf.placeholder(tf.float32,shape=[None,input_size])
y_input = tf.placeholder(tf.float32,shape=[None,no_classes])
#定义具有输入大小和分类数量的形状的权重变量
weights = tf.Variable(tf.random_normal([input_size,no_classes]))
bias = tf.Variable(tf.random_normal([no_classes]))
logits = tf.matmul(x_input,weights) + bias#(1,input_size)X(input_size,no_classes)+(1,no_classes)=(1,no_classes)
softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_input,logits=logits)#把(samples,no_classes)每一行作softmax，然后与每一行真实标签作交叉熵
loss_operation = tf.reduce_mean(softmax_cross_entropy)#从样本维上作平均
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss_operation)

session = tf.Session()
session.run(tf.global_variables_initializer())

for batch_no in range(total_batches):
    mnist_batch = mnist_data.train.next_batch(batch_size)
    _,loss_value = session.run([optimiser,loss_operation],feed_dict={x_input:mnist_batch[0],y_input:mnist_batch[1]})
    print("损失值:",loss_value)

signature_defination(session,"./models/case1_on_minist/",x_input,logits)
predictions = tf.argmax(logits,1)#因为每一行是对一个样本的预测概率分布，所以axis参数值为1就是找出一行中最大值的位置
correct_predictions = tf.equal(predictions,tf.argmax(y_input,1))#equal是将对应位置元素逐个比较。这里概率最大值的位置和真实one_hot标签1所在的位置一致那么就算是预测正确则返回True，否则返回False
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions,tf.float32))#先将True、False转为1.0，0.0，然后计算平均值即为准确率
test_images,test_labels = mnist_data.test.images,mnist_data.test.labels
accuracy_value = session.run(accuracy_operation,feed_dict={x_input:test_images,y_input:test_labels})
print(session.run(correct_predictions,feed_dict={x_input:test_images,y_input:test_labels}))
print("测试的准确率是:",accuracy_value)
session.close()


