"""
@Time : 2021/5/31 15:28
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
import numpy as np
from utility.model_process import universal_image_process
modelpath1 = "./models/dcgan"
modelpath2 = "./models/yolo"

import os
print(os.path.exists(modelpath1))

def method1():
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],modelpath1)#被加载的图就是当前的默认图
        #从下面的代码形式中可以猜测meta_graph.signature_def["xxx"]返回的是一个对象
        #一般的输入输出key是input_x和outputs。GAN的是input_z和outputs_Gz。
        input_tensor_name = meta_graph.signature_def["test_signature"].inputs["input_z"].name
        output_tensor_name = meta_graph.signature_def["test_signature"].outputs["outputs_Gz"].name
        print("输入输出张量是:")
        print(input_tensor_name,output_tensor_name)
        graph = tf.get_default_graph()

        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)
        print(input_tensor.shape)
        shape = [int(input_tensor.shape[1])]
        print(type(shape),shape)
        image = universal_image_process("./testimage.jpg",shape)
        result = sess.run(output_tensor,feed_dict={input_tensor:image})
        # print(result)

def for_yolo():
    modelpath = "./models/yolo"
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], modelpath)
        input_img_name = meta_graph.signature_def["prediction_signature"].inputs['input_img'].name
        input_img_shape_name = meta_graph.signature_def["prediction_signature"].inputs["input_image_shape"].name
        print(input_img_name)
        graph = tf.get_default_graph()
        input_img_tensor = graph.get_tensor_by_name(input_img_name)
        input_img_shape_tensor = graph.get_tensor_by_name(input_img_shape_name)
        print(input_img_tensor.shape)
        print(input_img_shape_tensor.shape)
for_yolo()