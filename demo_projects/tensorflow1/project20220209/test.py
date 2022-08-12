import os
# os.system("conda activate tf2.0 && python -m tf2onnx.convert --saved-model %s --output %s"%("./models/flowers_cli_savedmodel","./models/flowers.onnx"))
# from 图片处理 import get_images_labels
# import tensorflow.compat.v1 as tf
#
# def modify(frozen_graph_def):
#     for node in frozen_graph_def.node:
#         if node.op == 'RefSwitch':
#             print('RefSwitch')
#             node.op = 'Switch'
#             for index in range(len(node.input)):
#                 if 'moving_' in node.input[index]:
#                     node.input[index] = node.input[index] + '/read'
#         elif node.op == 'AssignSub':
#             print('AssignSub')
#             node.op = 'Sub'
#             if 'use_locking' in node.attr: del node.attr['use_locking']
#         elif node.op == 'AssignAdd':
#             print('AssignAdd')
#             node.op = 'Add'
#             if 'use_locking' in node.attr: del node.attr['use_locking']
#         elif node.op == 'Assign':
#             print('Assign')
#             node.op = 'Identity'
#             if 'use_locking' in node.attr: del node.attr['use_locking']
#             if 'validate_shape' in node.attr: del node.attr['validate_shape']
#             if len(node.input) == 2:
#                 # input0: ref: Should be from a Variable node. May be uninitialized.
#                 # input1: value: The value to be assigned to the variable.
#                 node.input[0] = node.input[1]
#                 del node.input[1]
# print("===========获取预测数据================")
# images,labels=get_images_labels(3)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     with open("./models/classification_test.pb", 'rb') as f:
#         graph_def = tf.GraphDef()  # 序列化的图对象，可以接收序列化数据形成还原图的信息
#         graph_def.ParseFromString(f.read())
#         tf.get_default_graph()
#         inputtensor,outputtensor = tf.import_graph_def(graph_def,return_elements=["inputtest:0","outputtest:0"])
#         print("预测结果:",sess.run(outputtensor,feed_dict={inputtensor:images}))
#         modify(graph_def)
#         builder = tf.saved_model.builder.SavedModelBuilder("./models/flowers_cli_savedmodel")
#         signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={"inputtest": inputtensor},
#                                                               outputs={"outputtest": outputtensor})
#         builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
#                                              signature_def_map={"test_signature": signature})
#         builder.save()

#===========================================================
# import onnxruntime
# import numpy as np
# model = onnxruntime.InferenceSession("./models/flowers.onnx")
# inputs = model.get_inputs()[0].name
# outputs = model.get_outputs()[0].name
# shape = model.get_inputs()[0].shape[1:]
# shape_withbatch = [1]
# shape_withbatch.extend(shape)
# data = np.random.randn(*shape_withbatch).astype("float32")
# result = model.run([outputs],{inputs:data})
# print(result[0])
import numpy as np
array = np.ones((4,4,2)).reshape([2,4,4])
a = [[0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,1]]
a = np.sum(a,axis=1).astype("float")
print(a)
