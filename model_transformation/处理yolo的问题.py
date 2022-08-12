"""
@Time : 2021/7/13 15:34
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
import os
from tensorflow.saved_model import signature_constants, signature_def_utils, tag_constants, utils

yolopath = "./models/yolo"
yolo_pb = "./models/pbmodels/yolo.pb"
yolo_onnx = "./models/onnxmodels/yolo.onnx"
yolo_pb_modified = "./models/pbmodels/yolo_modified.pb"
yolo_sm_modified = "./models/yolo_modified"

def modify(frozen_graph_def):
    for node in frozen_graph_def.node:
        if node.op == 'RefSwitch':
            print('RefSwitch')
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            print('AssignSub')
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'AssignAdd':
            print('AssignAdd')
            node.op = 'Add'
            if 'use_locking' in node.attr: del node.attr['use_locking']
        elif node.op == 'Assign':
            print('Assign')
            node.op = 'Identity'
            if 'use_locking' in node.attr: del node.attr['use_locking']
            if 'validate_shape' in node.attr: del node.attr['validate_shape']
            if len(node.input) == 2:
                # input0: ref: Should be from a Variable node. May be uninitialized.
                # input1: value: The value to be assigned to the variable.
                node.input[0] = node.input[1]
                del node.input[1]

def saved_as_pb(sess,savepath,outputname:list=None):
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,outputname)
    with tf.gfile.FastGFile(savepath, mode='wb') as f:
        f.write(frozen_graph_def.SerializeToString())

def load_yolo(modelpath,pbpath):
    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], modelpath)

        print("prediction_signature" in dict(model.signature_def).keys())

        print("yolo模型")
        signature_def = model.signature_def["prediction_signature"]
        print("------------------------------------------------")
        boxes = signature_def.outputs["boxes"].name
        scores = signature_def.outputs["scores"].name
        classes = signature_def.outputs["classes"].name
        input_img = signature_def.inputs["input_img"].name
        input_image_shape = signature_def.inputs["input_image_shape"].name
        #concat_11:0 concat_12:0 concat_13:0 input_1:0 Placeholder_366:0
        outputnodes = [boxes[:-2],scores[:-2],classes[:-2]]
        saved_as_pb(sess,pbpath,outputnodes)
        return outputnodes

# load_yolo(yolopath,yolo_pb)

def load_pb_model_to_pb(modelpath,destinationpath,outputnodes):
    # graph_def = tf.GraphDef()
    # text_format.Merge(f.read(),graph_def)，这个是在保存的时候as_text=True时
    # graph_def.ParseFromString(f.read())，这个是在保存的时候as_text=False时
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess2:
        with tf.gfile.FastGFile(modelpath, 'rb') as f:

            graph_def = tf.GraphDef() #序列化的图对象，可以接收序列化数据形成还原图的信息
            graph_def.ParseFromString(f.read())
            modify(graph_def)
            tf.import_graph_def(graph_def,name="")
            saved_as_pb(sess2,savepath=destinationpath,outputname=outputnodes)
            print("-------------------------------------------------------------")

# load_pb_model_to_pb(yolo_pb,yolo_pb_modified,load_yolo(yolopath,yolo_pb))

def load_pb_model_to_savedmodel(modelpath,destinationpath,input_imgname,input_image_shapename,boxesname,scoresname,classesname):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        with tf.gfile.FastGFile(modelpath, 'rb') as f:
            graph_def = tf.GraphDef()  # 序列化的图对象，可以接收序列化数据形成还原图的信息
            graph_def.ParseFromString(f.read())
            modify(graph_def)
            print("-----------------------------")
            modify(graph_def)
            tf.import_graph_def(graph_def,name="")
            builder = tf.saved_model.builder.SavedModelBuilder(destinationpath)
            sess.run(tf.global_variables_initializer())
            input_img = tf.get_default_graph().get_tensor_by_name(input_imgname)
            input_image_shape = tf.get_default_graph().get_tensor_by_name(input_image_shapename)
            boxes = tf.get_default_graph().get_tensor_by_name(boxesname)
            scores = tf.get_default_graph().get_tensor_by_name(scoresname)
            classes = tf.get_default_graph().get_tensor_by_name(classesname)
            signature = signature_def_utils.predict_signature_def(inputs={"input_img": input_img,"input_image_shape":input_image_shape},
                                                                  outputs={"boxes":boxes,"scores":scores,"classes":classes})
            builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                 signature_def_map={"prediction_signature":signature})
            builder.save()

# load_pb_model_to_savedmodel(yolo_pb_modified,yolo_sm_modified,"input_1:0","Placeholder_366:0","concat_11:0","concat_12:0","concat_13:0")

def convert_savedmodel_into_onnx(modelpath, destinationpath, modelname=None):
    result = os.system("conda activate tensorflow-gpu && python -m tf2onnx.convert --saved-model %s --output %s" % (
        modelpath, destinationpath))
    print(result)

def convert_pb_into_onnx(modelpath,destinationpath,inputs=None,outputs=None):
    print(os.system("conda activate tensorflow-gpu && python -m tf2onnx.convert --input %s --inputs %s --outputs %s --output %s"%(modelpath,inputs,outputs,destinationpath)))

# load_yolo(yolopath)
# convert_savedmodel_into_onnx(yolo_sm_modified,yolo_onnx)
# os.system("conda activate tensorflow-gpu && python -m tf2onnx.convert --input ./models/pbmodels/yolo_modified.pb --inputs input_1:0,Placeholder_366:0 --outputs concat_11:0,concat_12:0,concat_13:0 --custom-ops Round,NonMaxSuppressionV3 --output ./models/onnxmodels/yolo.onnx")

# with tf.Session() as sess2:
#     with tf.gfile.FastGFile("./models/pbmodels/yolo.pb", 'rb') as f:
#         graph_def = tf.GraphDef() #序列化的图对象，可以接收序列化数据形成还原图的信息
#         graph_def.ParseFromString(f.read())
#         modify(graph_def)
#         tf.import_graph_def(graph_def,name="")

print(os.system("conda activate onnxtest && python -m tf2onnx.convert --saved-model ./models/yolov3 --opset 11 --output ./models/onnxmodels/yolo_v3.onnx"))