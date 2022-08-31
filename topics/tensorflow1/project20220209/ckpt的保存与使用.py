import tensorflow as tf
import numpy as np
#生成模拟数据
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + np.random.randn(*train_X.shape) * 0.3 # y=2x，但是加入了噪声

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
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)#Gradient descent

# 初始化变量
init = tf.global_variables_initializer()
# 训练参数
training_epochs = 10
display_step = 2
"""
 if len(self._last_checkpoints) > self.saver_def.max_to_keep:
      self._checkpoints_to_be_deleted.append(self._last_checkpoints.pop(0))
从源码可以看出，当保存次数大于max_to_keep指定的值时(默认为5),则新的文件被保存，但最早的将被
删除,比如这里当第6次保存文件的时候，lr-5将被保存下来，而lr-0将被删除，以保证最多保存5个检查点文件，所以这里
最终保存了lr-5到lr-9一共5个检查点文件(检查点文件是个抽象概念，不是指一个文件，上面已经说明每个检查点是通过三个文件来保存的)
"""
saver = tf.train.Saver()
"""
指定名称就行，不用加ckpt后缀，加了也会变成名称的一部分，而不是真正的后缀;
保存的原理是：名称加编号，比如lr-0就表示第一个检查点信息，其中每个检查点对应三个文件，例如lr-0.data(变量数值)、
lr-0.meta(计算图)、lr-0.index(计算图和变量值的对应)。
文件夹下名称为checkpoint的文本文件第一行记录了最新的检查点文件名称,以及从上到下从旧到新一行记录一个检查点名称,
tf也可以据此来控制检查点文件数量不超过设定值
"""
saved_dir = "models/ckptmodels1/lr"#所有的检查点及相关文件放在ckptmodels目录下，lr是给检查点文件起的名称，不用带后缀
def trainAndSave():
    # 启动session
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            for (x, y) in zip(train_X, train_Y):
                sess.run(optimizer, feed_dict={X: x, Y: y})
            # 显示训练中的详细信息
            if epoch % display_step == 0:
                loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
                print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
            # 保存模型,一个epoch保存一次
            """
            这里global_step的值指定为epoch，实际上这个数值本身没有意义，可以随意指定，作用只是为了区分每一步产生的检查点文件，
            因为若不区分，则每次产生的检查点文件都叫做lr.data、lr.index、lr.meta,那么后面的就把前面的覆盖了，最终checkpoint文本文件中
            也确实只记录了一个检查点信息
            """
            saver.save(sess, saved_dir,global_step=epoch)
        print("测试:")
        print("w:", sess.run(W), "b:", sess.run(b))
        print("所以当x的值是20时，y的值是:", sess.run(z, feed_dict={X: 20}))


def loadAndPredict():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, saved_dir+"-5")#指定具体要加载哪个检查点的模型
        print(sess.run(z, feed_dict={X: 20}))
        #使用自带工具获取最新的
        latest = tf.train.latest_checkpoint(saved_dir[:-3])#这里目录填./ckptmodels1，不要带检查点名称
        saver.restore(sess,latest)
        print(sess.run(z, feed_dict={X: 20}))
        #获取所有的检查点文件信息
        ckpt = tf.train.get_checkpoint_state(saved_dir[:-3])#同上
        print(ckpt)#可以看到ckpt.model_checkpoint_path属性保存的是最新的检查点文件
        saver.restore(sess,ckpt.model_checkpoint_path)
        print(sess.run(z, feed_dict={X: 20}))

# loadAndPredict()
# trainAndSave()

#打印模型信息
def print_model():
    from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
    print_tensors_in_checkpoint_file(saved_dir + "-4", None, True)
    print_tensors_in_checkpoint_file(saved_dir + "-3", None, True)

#获取检查点文件
def get_checkpoint():
    ckpt = tf.train.get_checkpoint_state(saved_dir)
    print(ckpt.model_checkpoint_path)

# get_checkpoint()