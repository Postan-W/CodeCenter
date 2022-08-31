import tensorflow as tf

a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)
sum = a + b #或者tf.add(a,b)
multiply = a * b #或者tf.multiply(a,b)
with tf.Session() as sess:
    sum_result = sess.run(sum,feed_dict={a:4,b:5})
    print(sum_result)
    multiply_result = sess.run(multiply,feed_dict={a:3,b:4})
    print(multiply_result)
    print(sess.run([sum,multiply],feed_dict={a:6,b:7}))
