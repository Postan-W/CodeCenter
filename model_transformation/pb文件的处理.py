"""
@Time : 2021/7/8 11:21
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
import subprocess
from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
import os
from tensorflow.python.util import compat
from tensorflow.saved_model import signature_constants, signature_def_utils, tag_constants, utils
import torch
#output/BiasAdd:0,Placeholder:0
modelpath = "./models/resnet18"
origin_pbpath = "./models/resnet18/saved_model.pb"
destination_pbpath = "./models/resnet18_modified/saved_model.pb"
origin_pbpath2 = "./models/googlenet/saved_model.pb"
modelpath2 = "./models/resnet18_modified"
destination_path2 = "./models/onnxmodels/resnet18.onnx"
modelpath20 = "./models/vggnet16"
pbmodel1 = "./models/pbmodels/resnet.pb"
pbmodel1_modified = "./models/pbmodels/resnet_modified.pb"
pb_to_onnx_path1 = "./models/onnxmodels/resnet.onnx"
pb_to_onnx_path2 = "./models/onnxmodels/resnet_modified.onnx"

#fc3_class_num/Softmax:0,Placeholder:0
vggnet16 = "./models/vggnet16"
vggnet16_pb = "./models/pbmodels/vggnet16.pb"
vggnet16_pb_modified = "./models/pbmodels/vggnet16_modified.pb"
vggnet16_savedmodel_modified = "./models/vggnet16_modified"
vggnet16_onnx = "./models/onnxmodels/vggnet16.onnx"
#fc_3/Softmax:0,Placeholder:0
vggnet19 = "./models/vggnet19"
vggnet19_pb = "./models/pbmodels/vggnet19.pb"
vggnet19_pb_modified = "./models/pbmodels/vggnet19_modified.pb"
vggnet19_savedmodel_modified = "./models/vggnet19_modified"

yolo = "./models/yolo"

def use_pbmodel(path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = path
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input = sess.graph.get_tensor_by_name("inputtest:0")
            output = sess.graph.get_tensor_by_name("outputtest:0")
            shape = [1]
            shape.extend(list(input.shape)[1:])
            print(shape)
            input_data = torch.randn(*shape)  # 输入要测试的数据，格式要一致
            predictions = sess.run(output, feed_dict={input: input_data})
            print("predictions:", predictions)
# use_pbmodel("conv_classification.pb")

def figure_out_the_input_output(modelpath):
    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], modelpath)
        signature_def = model.signature_def["test_signature"]
        print("------------------------------------------------")
        print(signature_def.outputs["outputs"].name)
        print(signature_def.outputs["outputs"].name[:-2])

        print(signature_def.inputs["input_x"].name)
        print(signature_def.inputs["input_x"].name[:-2])
        sess.close()
# figure_out_the_input_output(vggnet16)

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
    """
    把加载的ckpt模型保存为pb的关键点在于固化节点的操作时要指定输出节点名称。
    因为网络其实是比较复杂的，定义了输出结点的名字，那么freeze的时候就只把输出该结点所需要的子图都固化下来，其他无关的就舍弃掉。因为我们freeze模型的目的是接下来做预测。所以，output_node_names一般是网络模型最后一层输出的节点名称，或者说就是我们预测的目标。
    """
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,outputname)
    with tf.gfile.FastGFile(savepath, mode='wb') as f:
        f.write(frozen_graph_def.SerializeToString())

def load_savedmodel(savedmodel_path,pb_path):
    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],savedmodel_path)
        signature_def = model.signature_def["test_signature"]
        print("------------------------------------------------")
        output_tensor_name = signature_def.outputs["outputs"].name
        input_tensor_name = signature_def.inputs["input_x"].name
        saved_as_pb(sess,pb_path,[output_tensor_name[:-2]])
        return input_tensor_name,output_tensor_name

# load_savedmodel(vggnet16,vggnet16_pb)


def convert_savedmodel_into_onnx(modelpath,destinationpath,modelname=None):
        result = os.system("conda activate tensorflow-gpu && python -m tf2onnx.convert --saved-model %s --output %s" % (
        modelpath, destinationpath))
        print(result)



# convert_savedmodel_into_onnx(vggnet16, vggnet16_onnx)


#这个加载的是Pb模型文件,再保存为pb
def load_pb_model_to_pb(modelpath,destinationpath,output_tensor_name):
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
            saved_as_pb(sess2,savepath=destinationpath,outputname=[output_tensor_name[:-2]])
            print("-------------------------------------------------------------")

# sess2 = tf.Session()
# model = tf.saved_model.loader.load(sess2,[tf.saved_model.tag_constants.SERVING],vggnet16)
# sess2.close()
#
# output_tensor_name = "fc3_class_num/Softmax:0"
# load_pb_model_to_pb(vggnet16_pb,vggnet16_pb_modified,output_tensor_name)

#这个加载的是pb模型文件，保存为savedmodel
def load_pb_model_to_savedmodel(modelpath,destinationpath,inputname:str,outputname:str):
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
            input_tensor = tf.get_default_graph().get_tensor_by_name(inputname)
            output_tensor = tf.get_default_graph().get_tensor_by_name(outputname)
            signature = signature_def_utils.predict_signature_def(inputs={"input_x": input_tensor},
                                                                  outputs={"outputs":output_tensor})
            builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                 signature_def_map={"test_signature":signature})
            builder.save()





def convert_pb_into_onnx(modelpath,destinationpath,inputs=None,outputs=None):
    print(os.system("conda activate tensorflow-gpu && python -m tf2onnx.convert --input %s --inputs %s --outputs %s --output %s --custom-ops BatchNorm/AssignMovingAvg"%(modelpath,inputs,outputs,destinationpath)))

# convert_pb_into_onnx(pbmodel1_modified,pb_to_onnx_path2,"Placeholder:0","output/BiasAdd:0")


#这个加载的是savedmodel中的pb文件

def load_pb_savedmodel():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.gfile.FastGFile(origin_pbpath, 'rb') as f:
            data = compat.as_bytes(f.read())
            sm = saved_model_pb2.SavedModel()
            sm.ParseFromString(data)
            graph_def = sm.meta_graphs[0].graph_def
            print(sess.graph_def)
            tf.import_graph_def(graph_def)
            print(sess.graph_def)
            modify(graph_def)
            print("---------------------------------")
            modify(graph_def)
            print("---------------------------------------")
            with tf.gfile.FastGFile(destination_pbpath, mode='wb') as f:
                f.write(graph_def.SerializeToString())



def pipeline(inputsm,pbpath,modified_pbpath,modified_sm,onnxpath):
    """
    需要注意的是：每个步骤都开启了Session,则一定要为该Session指定图，否则第一个步骤生成的图变成了默认图，在后续步骤中就会使用到，从而引起错误
    """
    input_tensor_name,output_tensor_name = load_savedmodel(inputsm,pbpath)
    load_pb_model_to_pb(pbpath,modified_pbpath,output_tensor_name)
    load_pb_model_to_savedmodel(modified_pbpath,modified_sm,input_tensor_name,output_tensor_name)
    convert_savedmodel_into_onnx(modified_sm,onnxpath)

# pipeline(vggnet16,vggnet16_pb,vggnet16_pb_modified,vggnet16_savedmodel_modified,vggnet16_onnx)