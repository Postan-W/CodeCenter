from 图片处理 import get_images_labels
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
print("===========获取预测数据================")
images,labels=get_images_labels(3)
gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    with tf.gfile.FastGFile("./conv_classification.pb", 'rb') as f:
        graph_def = tf.GraphDef()  # 序列化的图对象，可以接收序列化数据形成还原图的信息
        graph_def.ParseFromString(f.read())
        tf.get_default_graph()
        inputtensor,outputtensor = tf.import_graph_def(graph_def,return_elements=["inputtest:0","outputtest:0"])
        print("预测结果:",sess.run(outputtensor,feed_dict={inputtensor:images}))
