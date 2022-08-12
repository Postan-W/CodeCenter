import tensorflow as tf

c1 = tf.constant([[1,2],[3,4]])
sess = tf.InteractiveSession()
c = tf.linspace(0.0,4.0,5)
print(c.eval())