"""
pb文件只保存了图结构，包含常量，不包含变量，解决方法是将变量variable转为常量constant
"""
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
saved_dir = "../test_code/models3_pb"

def saved_as_pb(sess):
    """
    sess: 当前使用的会话对象sess
    input_graph_def: 是一个GraphDef object ，及当前会话中的Graph
    output_node_names: graph输出节点的名称，形如 [“name1”，“name2” ]
    variable_names_whitelist: 需要转化的变量Variable所组成的list，默认情况下graph中的所有     variable均会转化成constant(by default, all variables are converted).
    variable_names_blacklist: 忽略转化。即不需要转化成constant的variables所组成的list
    """
    #var_list = tf.trainable_variables()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['add'])
    with tf.gfile.FastGFile(saved_dir+"/linearRegression.pb", mode='wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    """
    tf.train.write_graph(frozen_graph_def,saved_dir,"linearRegression2.pb",as_text=True)
    as_text=Fasle,等价于上面的保存方式，等到读取时的方法为graph_def.ParseFromString(f.read())；
    as_text=True,等到读取时的方法为text_format.Merge(f.read(),graph_def)。
    
    参见pb模型的载入与使用
    """


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
        saved_as_pb(sess)
        print("测试:")
        print("w:", sess.run(W), "b:", sess.run(b))
        print("所以当x的值是20时，y的值是:", sess.run(z, feed_dict={X: 20}))

trainAndSave()

"""
注：可以用如下方式将ckpt或savedmodel转为pb
with tf.Session(config = config) as sess:
    # 构建模型结构   
    saver = tf.train.import_meta_graph('./ckpt_model/keypoint_model.ckpt-99.meta')
    # 载入模型参数
    saver.restore(sess,'./ckpt_model/keypoint_model.ckpt-99')
    
    # 将变量转化成constant
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ['dense4_output'])#前提是知道输出操作的名称
    tf.io.write_graph(frozen_graph_def,"pb_model","freeze_eval_graph.pb",as_text=False)
    print("model have been frozen... ...")
"""