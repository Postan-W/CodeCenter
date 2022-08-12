import tensorflow as tf
print(tf.__version__)

x1 =tf.constant(12)
x2 = tf.constant(13)
x3 =tf.constant(12,name="test")
x4 =tf.constant(12,name="test")
x5 = tf.Variable(100)
x6 = tf.Variable(100)
x7 = tf.Variable(100,name="test")
x8 = tf.Variable(100,name="test")
sum = x1 + x2
print(sum)
sum2 = x5 + x6
print(sum2)
print(x1)
print(x2)
print(x3)
print(x4)
print(x5)
print(x6)
print(x7)
print(x8)
with tf.name_scope("outer"):
    x9 = tf.Variable(100, name="test")
    x10 = tf.Variable(100, name="test")
    print(x9)
    print(x10)
    with tf.name_scope("inner"):
        x11 = tf.Variable(100, name="test")
        x12 = tf.Variable(100, name="test")
        print(x11)
        print(x12)
print(tf.get_default_graph().get_operations())
print(sum)

with tf.Session() as sess:
    print(sess.run(sum))