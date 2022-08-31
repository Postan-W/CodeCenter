"""
:n，表示返回的第n+1个张量；一般一个动作只返回一个张量，所以一般看到的都是:0。如下面的firstvar:0
"""
import tensorflow as tf

def without_scope():
    tf.reset_default_graph()
    var1 = tf.Variable(1.0, name='firstvar')
    print("var1:", var1.name)
    var1 = tf.Variable(2.0, name='firstvar')  # tf变量标识同名则会被tf修正为_加编号,firstvar_1
    print("var1:", var1.name)
    var2 = tf.Variable(3.0)  # 没有给变量指定标识则默认是Variable_编号
    print("var2:", var2.name)
    var2 = tf.Variable(4.0)
    print("var1:", var2.name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("var1=", var1.eval())
        print("var2=", var2.eval())

    #
    get_var1 = tf.get_variable("firstvar", [1], initializer=tf.constant_initializer(0.3))
    print("get_var1:", get_var1.name)  # 前面已经有两个firstvar,所以这一个是firstvar_2

    get_var1 = tf.get_variable("firstvar1", [1], initializer=tf.constant_initializer(0.4))
    print("get_var1:", get_var1.name)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("get_var1=", get_var1.eval())

"""
带作用域创建变量。作用域可以嵌套，在不同作用域中get_variable可以创建同名变量，否则会报错。
tf.Variable每次都创建新的变量，而get_variable有共享机制。
父域的初始化值也可以被子域继承，一个域中的变量没有指定初始化值的时候则会使用域定义时的初始化值
"""
def with_scope():
    tf.reset_default_graph()#将现有的图清空，也就是图里面的变量清空
    with tf.variable_scope("test1", reuse=tf.AUTO_REUSE,initializer=tf.constant_initializer(0.9)):#表示支持共享变量
        var1 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32,initializer=tf.constant_initializer(0.5))
        """
        因为名称相同，所以会共享上面get_variable创建的变量。注意形状和类型参数必须保持一致，否则会报错,而initializer参数可以继承。
        实际上除了名称，其他参数都可以省略，如下面的var2。共享相当于两个图节点共用一个数据(一个内存地址),下面如果
        令var2=var1,输出的结果会不变，但与共享变量机制的区别是，此时只有一个图节点
        """
        var2 = tf.get_variable("firstvar")#共享变量
        test1 = tf.get_variable("test1", shape=[2], dtype=tf.float32)
        with tf.variable_scope("test2"):#父域声明过共享了，父域内的所有域也会共享
            var3 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32,initializer=tf.constant_initializer(0.6))
            var4 = tf.get_variable("firstvar", shape=[2], dtype=tf.float32)#共享变量
            test2 = tf.get_variable("test2", shape=[2], dtype=tf.float32)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("var1value={},var1name={}".format(var1.eval(),var1.name))
        print("var2value={},var2name={}".format(var2.eval(), var2.name))
        print("var3value={},var3name={}".format(var3.eval(), var3.name))
        print("var4value={},var4name={}".format(var4.eval(), var4.name))
        print("test1value={},test1name={}".format(test1.eval(), test1.name))
        print("test2value={},test2name={}".format(test2.eval(), test2.name))

#灵活地指明作用域；as语句
def use_scope():
    with tf.variable_scope("scope1") as sp:  # 使用as语句定义了sp,这个sp就可以被当做作用域拿去用
        var1 = tf.get_variable("v", [1])
    print("sp:", sp.name)
    print("var1:", var1.name)
    with tf.variable_scope("scope2"):
        var2 = tf.get_variable("v", [1])
        with tf.variable_scope(sp) as sp1:
            var3 = tf.get_variable("v3", [1])  # 尽管在scope2下面，但因为指定了scope为sp，所以var3在作用域scope1下面，而不在scope2下面

            with tf.variable_scope(sp1):  # sp1也被拿来用
                var4 = tf.get_variable("v4", [1])
    print("sp1:", sp1.name)
    print("var2:", var2.name)
    print("var3:", var3.name)
    print("var4:", var4.name)
#name_scope只限制op,而variable_scope既可以限制变量也可以限制op
def use_namescope():
    with tf.variable_scope("scope"):
        with tf.name_scope("bar"):
            v = tf.get_variable("v", [1])#没有受到bar的限制
            x = 1.0 + v
            """
            与variable_scope名称为空串那么该层作用域就是一个空串，而name_scope则会将作用域返回到顶层;
            也就是说这里的空串会将上层的scope、bar两个作用域都去掉，当前域的op置于顶层
            """
            with tf.name_scope(""):
                y = 1.0 + v
    print("v:", v.name)
    print("x.op:", x.op.name)
    print("y.op:", y.op.name)
use_namescope()