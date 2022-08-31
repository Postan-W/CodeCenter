import numpy as np
import tensorflow as tf
print("==========================图的创建与使用====================================================")
c = tf.constant(0.0)#c是在默认图中
g = tf.Graph()#创建了一个图对象
with g.as_default():
  c1 = tf.constant(0.0)#在g中而不在默认图中
  print(c1.graph)
  print(g)
  print(c.graph)

c2 = tf.constant(1.0)
print(c2.graph)
g2 = tf.get_default_graph()#获取的是默认图，也就是上面c和c2所在的图
print(g2)

"""
重置默认图，也就是说上面c、c2所在的图被清除了而被新的默认图代替，前提是之前的默认图的资源已经被释放了，比如使用
该图的会话已经结束。
"""
tf.reset_default_graph()
g3 =  tf.get_default_graph()
print(g3)#默认图是新的，不是上面c、c2所在的那个了
c3 = tf.constant(2.0)
print(c3.graph)

print("========================从图中获取======================================================")
print("-----------------获取张量-----------------")
print(c1.name)
t = g.get_tensor_by_name(name="Const:0")
print(t)
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[1.0], [3.0]])
tensor1 = tf.matmul(a, b, name='exampleop')
print(tensor1.name, tensor1)
test = g3.get_tensor_by_name("exampleop:0")#获取的就是tensor1
print(test)
print("------------------------获取操作---------------------------")
print(tensor1.op.name)
testop = g3.get_operation_by_name("exampleop")
print(testop)
with tf.Session() as sess:
    test = sess.run(test)#运行操作，返回结果
    print(test)
    test = tf.get_default_graph().get_tensor_by_name("exampleop:0")
    print(test)

print("---------获取操作节点列表------------")
with g.as_default():
    c4 = tf.constant(2.0)
    c5 = tf.constant(3.0)
operations = g.get_operations()#将返回一个包含const,const_1,const_2的列表
print(operations)

#使用as_grpaph_element()；没明白有什么意义
def test():
    print(c1)
    test = g.as_graph_element(c1)
    print(test)
    print(test==c1)#结果是True

print("===========================默认图的问题========================")
test = tf.constant(5.0)#在全局中创建的，test是在默认图中
print(test.graph)
print(tf.get_default_graph())
with tf.Graph().as_default():
    print(tf.get_default_graph())
print(tf.get_default_graph())
"""
上面打印了4个图对象，1,2,4都是一样的，为全局默认图；3是在tf.Graph()创建的图的作用域内，所以默认图不同于全局的，其实就是
tf.Graph()创建的图，所以在这个作用域内创建节点都将被加到这个图中
"""

































  