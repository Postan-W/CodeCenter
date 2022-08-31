
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

plotdata = { "batchsize":[], "loss":[] }
def moving_average(a, w=10):
    if len(a) < w:
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]


#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声
#显示模拟数据点
# plt.plot(train_X, train_Y, 'ro', label='Original data')
# plt.legend()
# plt.show()

# 创建模型
# 占位符
X = tf.placeholder("float",name="model_input")
Y = tf.placeholder("float",name="real_value")
# 模型参数
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.zeros([1]), name="bias")

# 前向结构
z = tf.multiply(X, W) + b
print(X,Y,z)

#反向优化
cost =tf.reduce_mean(tf.square(Y - z))
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #Gradient descent

# 初始化变量
init = tf.global_variables_initializer()
# 训练参数
training_epochs = 10
display_step = 2

saver = tf.train.Saver()
saved_dir = "../test_code/models2_savedmodel"#这个路径不能已经存在

#在tf1中有两种保存savedmodel的方式，这里试验使用signature的方式
from tensorflow.saved_model import signature_constants, signature_def_utils, tag_constants, utils

def signature_defination(sess,saved_dir):
    #存储了整个graph的信息，并且由于定义了signature，所以又指定了如输入输出这样的信息
    builder = tf.saved_model.builder.SavedModelBuilder(saved_dir)
    signature = signature_def_utils.predict_signature_def(inputs={"model_input":X},
                                                          outputs={"model_output":z})
    #注：通过加载的代码可以看到，上面的inputs和outputs将作为meta_graph.signature_def["graph_information"]返回的signature对象的属性
    #tags的作用是为了区分图，通常这个名称 tag 以该计算图的功能和使用到的设备命名，比如 serving or training， CPU or GPU。一个图add一次
    builder.add_meta_graph_and_variables(sess=sess,tags=[tag_constants.SERVING],signature_def_map={'graph_information':signature})
    builder.save()

def trainAndSave():
    # 启动session
    with tf.Session() as sess:
        sess.run(init)

        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # 显示训练中的详细信息
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
                if not (loss == "NA"):
                    plotdata["batchsize"].append(epoch)
                    plotdata["loss"].append(loss)

        signature_defination(sess,saved_dir)
        print("测试:")
        print("w:", sess.run(W), "b:", sess.run(b))
        print("所以当x的值是20时，y的值是:", sess.run(z, feed_dict={X: 20}))



trainAndSave()