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
tf.summary.histogram("z",z)#将z以直方图形式展示
print(X,Y,z)
#反向优化
cost =tf.reduce_mean(tf.square(Y - z))
tf.summary.scalar("cost",cost)#将损失值以标量形式展示
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# 初始化变量
init = tf.global_variables_initializer()
# 训练参数
training_epochs = 10
display_step = 2
#在命令行中使用tensorboard --logdir 该目录启动服务，根据服务提示地址访问
tensorboard_dir = "tensorboard_logs/logs1"
with tf.Session() as sess:
    sess.run(init)
    merged_summary = tf.summary.merge_all()#合并所有summary，用于下面一同操作，一同保存
    summary_writer = tf.summary.FileWriter(tensorboard_dir,sess.graph)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})
        """
        这里是每个epoch保存一次，于是每个epoch(即横坐标)就对应一个点，最终形成折线图比较直观
        """
        summary_str = sess.run(merged_summary, feed_dict={X: x, Y: y})
        summary_writer.add_summary(summary_str, epoch)
        # 显示训练中的详细信息
        if epoch % display_step == 0:
            loss = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", epoch + 1, "cost=", loss, "W=", sess.run(W), "b=", sess.run(b))
