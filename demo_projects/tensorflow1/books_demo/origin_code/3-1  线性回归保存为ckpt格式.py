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
training_epochs = 5
display_step = 2
#注意查看Saver源代码可知max_to_keep=5,要想保存更多检查点模型，在参数中指定
saver = tf.train.Saver()
saved_dir = "../test_code/models1_ckpt/lr"
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
            # 保存模型,一个epoch保存一次
            saver.save(sess, saved_dir, global_step=epoch)
        print("测试:")
        print("w:", sess.run(W), "b:", sess.run(b))
        print("所以当x的值是20时，y的值是:", sess.run(z, feed_dict={X: 20}))



def loadAndPredict():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, saved_dir)
        print(sess.run(z, feed_dict={X: 20}))

# trainAndSave()

#打印模型信息
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
print_tensors_in_checkpoint_file(saved_dir+"-4",None,True)
print_tensors_in_checkpoint_file(saved_dir+"-3",None,True)