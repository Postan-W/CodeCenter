"""
需要额外的库包含：
keras 2.2.4
keras2onnx 1.7.0
onnx 1.5.0
onnxruntime 1.6.0
py4j 0.10.9.1
tf2onnx 1.5.1
savedmodel转换模块会自动创建目录，目录不能已存在。h5转换模块相反
"""
import tensorflow as tf
from tensorflow.saved_model import signature_constants, signature_def_utils, tag_constants, utils
import argparse
import sys
sys.path.append("AI/")
sys.path.append("AI/ModelTransformation/")
import hdfs
import os
from hdfs.ext.kerberos import KerberosClient
from config.config import *
import logging
import keras
import keras2onnx
import subprocess
import onnx
import re
from ComputerVision.dl_config import *
import logging
import shutil
import time
print("展示镜像内特定的python库")
print(os.system("pip list |grep -E 'tensorflow|keras|onnx|tf2onnx|h5py|protobuf'"))

"""
流上的resnet18,vggnet16,vggnet19存在batchnorm节点，而onnx无法识别这种节点，为了处理这种错误，现有的解决方案是将原模型的savedmodel转为pb,
加载pb得到grap_def,对graph_def进行处理后，保存为pb,将保存的pb再转回为savedmodel，最终将这个savedmodel转为onnx。
即savedmodel->pb->pb->savedmodel->onnx
"""

#修改节点
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

#将会话信息保存为pb
def saved_as_pb(sess, savepath, outputname: list = None):
    """
    把加载的ckpt模型保存为pb的关键点在于固化节点的操作时要指定输出节点名称。
    因为网络其实是比较复杂的，定义了输出结点的名字，那么freeze的时候就只把输出该结点所需要的子图都固化下来，其他无关的就舍弃掉。因为我们freeze模型的目的是接下来做预测。所以，output_node_names一般是网络模型最后一层输出的节点名称，或者说就是我们预测的目标。
    """
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, outputname)
    with tf.gfile.FastGFile(savepath, mode='wb') as f:
        f.write(frozen_graph_def.SerializeToString())

#将会转换出错的savedmodel模型暂时转为pb
def load_savedmodel(savedmodel_path,pb_path):
    with tf.Session() as sess:
        model = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],savedmodel_path)
        signature_def = model.signature_def["test_signature"]
        print("------------------------------------------------")
        output_tensor_name = signature_def.outputs["outputs"].name
        input_tensor_name = signature_def.inputs["input_x"].name
        saved_as_pb(sess,pb_path,[output_tensor_name[:-2]])
        return input_tensor_name,output_tensor_name

#将包含onnx不能识别节点的pb模型修改后保存为pb模型
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

#将修改后的pb模型转为sm
def load_pb_model_to_savedmodel(modelpath, destinationpath, inputname: str, outputname: str):
    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        with tf.gfile.FastGFile(modelpath, 'rb') as f:
            graph_def = tf.GraphDef()  # 序列化的图对象，可以接收序列化数据形成还原图的信息
            graph_def.ParseFromString(f.read())
            modify(graph_def)
            print("-----------------------------")
            modify(graph_def)
            tf.import_graph_def(graph_def, name="")
            builder = tf.saved_model.builder.SavedModelBuilder(destinationpath)
            sess.run(tf.global_variables_initializer())
            input_tensor = tf.get_default_graph().get_tensor_by_name(inputname)
            output_tensor = tf.get_default_graph().get_tensor_by_name(outputname)
            signature = signature_def_utils.predict_signature_def(inputs={"input_x": input_tensor},
                                                                  outputs={"outputs": output_tensor})
            builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING],
                                                 signature_def_map={"test_signature": signature})
            builder.save()


#处理一下路径的问题
def path_generator(inputsm,onnxpath):
    pbpath_parent = inputsm + "pb"
    os.mkdir(pbpath_parent)
    pbpath = pbpath_parent + "/origin.pb"
    modified_pbpath = pbpath_parent + "/modified.pb"
    modified_sm = inputsm + "modified"

    return inputsm,pbpath,modified_pbpath,modified_sm,onnxpath,pbpath_parent


#整合上面的转化过程
def pipeline(inputsm,pbpath,modified_pbpath,modified_sm,onnxpath,pbpath_parent):
    """
    需要注意的是：每个步骤都开启了Session,则一定要为该Session指定图，否则第一个步骤生成的图变成了默认图，在后续步骤中就会使用到，从而引起错误
    """
    print("----------------------stage1:sm>>>>>>pb---------------------------------")
    input_tensor_name,output_tensor_name = load_savedmodel(inputsm,pbpath)
    print("----------------------stage2:pb>>>>>>pb-----------------------------------")
    load_pb_model_to_pb(pbpath,modified_pbpath,output_tensor_name)
    print("----------------------stage3:pb>>>>>>sm-----------------------------------")
    load_pb_model_to_savedmodel(modified_pbpath,modified_sm,input_tensor_name,output_tensor_name)
    print("----------------------stage4:sm>>>>>>onnx----------------------------------")
    convert_savedmodel_into_onnx(modified_sm,onnxpath)

    print("----------------------stage5:清理文件----------------------------------------")
    shutil.rmtree(pbpath_parent)
    shutil.rmtree(inputsm)



#从传来的模型路径中拿出模型名称，以此作为onnx模型的名称
def get_model_name(target_path:str,modeltype:str=None)->str:
    #如果hdfs模型路径上存在version字样，那么version前面的部分即是模型名称，使用正则表达式取出该名称。如果不存在，那么给该模型取个默认名称
    pattern = re.compile("(?<=/).+(?=/version)")
    try:
        result = pattern.search(target_path).group()  # 这一步得到的是包括名称以及前面的部分
        result = os.path.split(result)[1]
        print("得到的名称是：",result)
        result = result + ".onnx"
    except:
        result = time.asctime(time.localtime(time.time())).replace(" ","")
        result = modeltype + "_" + result + ".onnx"

    return result

def find_latest_meta(parent_path:str)->str:
    file_list = os.listdir(parent_path)
    count = 0
    max = 0
    meta_file = ""
    for file in file_list:
        if file.endswith(".meta"):
            count += 1

    if count == 1:
        for file in file_list:
            if file.endswith(".meta"):
                meta_file = parent_path + file if parent_path.endswith("/") else parent_path +"/" + file
                break
    else:
        for file in file_list:
            if file.endswith(".meta"):
                number = int(os.path.splitext(file)[0][-1])
                max = number if max < number else max

        for file in file_list:
            if file.endswith(".meta"):
                if int(os.path.splitext(file)[0][-1]) == max:
                    meta_file = parent_path + file if parent_path.endswith("/") else parent_path + "/" + file
                    break

    return meta_file

def check_type(files:list)->str:
    modelType = ""
    for file in files:
        if file.endswith(".pb") and len(files) > 1:
            modelType = "savedmodel"
            print("这是savedmodel模型")
            break
        elif file.endswith(".h5"):
            modelType = "h5"
            print("这是h5模型")
            break
        elif file.endswith(".pb") and len(files) == 1:
            modelType = "pb"
            print("这是pb模型")
            break
        elif file == "checkpoint":
            modelType = "ckpt"
            print("这是ckpt模型")
            break
    return modelType

#savedmodel->onnx模型不需要指定输入输出
def convert_savedmodel_into_onnx(modelpath,destinationpath):
    #tf1.12.0&onnx==1.5.0&tf2onnx==1.5.1&--opset 7
    #tf1.11&onnx==1.9.0&tf2onnx==1.8.5&--opset 9
    #流上产生的yolov3模型对应的opset为11
    print("开始转换savedmodel模型")
    result = os.system("python -m tf2onnx.convert --saved-model %s --output %s"%(modelpath,destinationpath))
    #yolov3对应的opset为11
    if result != 0:
        result = os.system("python -m tf2onnx.convert --saved-model %s --opset 11 --output %s"%(modelpath,destinationpath))
    print("os.system结果:", result)
    return result


def convert_h5_into_onnx(modelpath,destinationpath):
    # h5model = tf.keras.models.load_model(modelpath)
    #低版本的tensorflow集成的keras使用该句会出错KeyError: 'weighted_metrics'。这里换做keras自身，而不是集成的
    try:
        h5file = ""
        for file in os.listdir(modelpath):
            if file.endswith(".h5"):
                h5file = file
                break
        print("开始转换h5模型")
        entire_path = modelpath+"/"+h5file
        print("源地址是:",entire_path)
        print("创建文件夹(0表示正常):",os.system("mkdir -p %s"%(os.path.split(destinationpath)[0])))
        h5model = keras.models.load_model(entire_path)
        onnxmodel = keras2onnx.convert_keras(h5model, h5model.name)
        onnx.save_model(onnxmodel, destinationpath)
    except Exception as error:
        print("转换h5出现的问题是:",error)

def convert_ckpt_into_onnx(modelpath,destinationpath,inputs:str=None,outputs:str=None):
    latest_meta_file = find_latest_meta(modelpath)
    latest_ckpt = os.path.splitext(latest_meta_file)[0]
    print(latest_meta_file)
    print("转换返回状态(0代表正常):",os.system(
        "python -m tf2onnx.convert --checkpoint %s --inputs %s --outputs %s --output %s"%(
        latest_meta_file, inputs, outputs, destinationpath)))
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph(latest_meta_file)
    #     # 载入的是最新的那个ckpt文件，也就是数字最大的
    #     saver.restore(sess, latest_ckpt)
    #     graph = tf.get_default_graph()  # 获取当前图
    #     print([tensor.name for tensor in graph.as_graph_def().node])
    #     # saved_as_pb(sess,destinationpath,ckpt_outputs)

def convert_pb_into_onnx(modelpath,destinationpath,inputs=None,outputs=None):
    print("转换返回状态(0代表正常):",os.system("python -m tf2onnx.convert --input %s --inputs %s --outputs %s --output %s"%(modelpath,inputs,outputs,destinationpath)))

def convert(modelType:str,modelpath,destinationpath):
    if modelType == "h5":
        convert_h5_into_onnx(modelpath,destinationpath)
    elif modelpath == "pb":
        convert_pb_into_onnx(modelpath,destinationpath)
    elif modelType == "savedmodel":
        result = convert_savedmodel_into_onnx(modelpath, destinationpath)
        if result == 0:
            print("---------------------------转换成功-------------------------------------")
        else:
            print("--------------------------转换失败---------------------------------")
            print("---------------------------下面开始处理------------------------------")
            pipeline(*path_generator(modelpath,destinationpath))
    elif modelType == "ckpt":
        convert_ckpt_into_onnx(modelpath,destinationpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='the path to get the model and place the model after being converted')
    parser.add_argument('--model_path', required=True, type=str, help='the path to download the model')
    parser.add_argument('--target_path', required=True, type=str, help='the path to place the model which is the converted version'
                                                                     'of the model from the sourcePath')
    parser.add_argument('--keytab_path', required=True, type=str,
                        help='to pass the certification of kerberos')

    args = parser.parse_args()

    sourcePath = args.model_path + "/model"
    targetPath = args.target_path
    keytab_path = args.keytab_path

    client = kerberos_hdfs_connect(keytab_path)
    print("kerberos认证通过")
    #为每一个模型设定一个唯一的存放路径source_local
    source_local = time.asctime(time.localtime(time.time())).replace(" ","")
    client.download(sourcePath,source_local,overwrite=True)
    print("模型下载成功，from {0} to {1}".format(sourcePath,source_local))
    local_files = os.listdir(source_local)
    print("本地保存的模型文件为:",local_files)


    files = os.listdir(source_local)
    print("清理过后,本地保存的模型文件为:", files)
    modelType = check_type(files)
    # 下载下来的模型是把模型文件放到model目录中打包的，这里再添加一层model目录形成两层，这样符合平台模型服务等功能解包模型的层次结构
    destination_local = source_local + "destination" + "/model"

    modelName = get_model_name(targetPath,modelType)
    print("模型本地保存路径是:",destination_local+"/"+modelName)
    convert(modelType,source_local,destination_local+"/"+modelName)
    print("模型转化成功，from {0} to {1}".format(source_local, destination_local))
    print("转换完的模型文件为:",os.listdir(destination_local))
    client.upload(targetPath,destination_local[:-6],overwrite=True)#将model目录也传过去
    print("模型上传成功，from {0} to {1}".format(destination_local+"/"+modelName,targetPath))
