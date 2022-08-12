"""
@Time : 2021/5/31 9:24
@Author : wmingzhu
@Annotation : 参考链接：https://bindog.github.io/blog/2020/03/13/deep-learning-model-convert-and-depoly/
参考链接：https://www.it610.com/article/1290365725277495296.htm
"""
#需要使用tf2onnx模块，使用命令行方式
#savedmodel转换模块会自动创建目录，目录不能已存在。h5转换模块相反
#目前验证发现不需要提供--tag、--signature_def、--inputs、--outputs等参数也能成功加载模型并预测
modelpath1 = "./LeNet"
destination1 = "./onnxmodels/LeNetNoOpset.onnx"
modelpath2 = "./case2_CNN"
destination2 = "./onnxmodels/case2_CNN.onnx"
modelpath3 = "./model_miniVGGNet.h5"
destination3 = "./onnxmodels/miniVGGNet.onnx"
modelpath4 = "./model_MobileNet_v1.h5"
destination4 = "./onnxmodels/mobileNet_v1.onnx"
modelpath5 = "./models/model_MobileNet_v2.h5"
destination5 = "./models/onnxmodels/MobileNet_v2.onnx"
modelpath6 = "./models/yolo"
destination6 = "./models/onnxmodels/yolo.onnx"
modelpath7 = "./dcgan"
destination7 = "./onnxmodels/dcgan.onnx"
modelpath8 = "./models/alexnet"
destination8 = "./models/onnxmodels/alexnet.onnx"
modelpath9 = "./models/inception3"
destination9 = "./models/onnxmodels/inception3.onnx"
modelpath10 = "./models/inception4"
destination10 = "./models/onnxmodels/inception4.onnx"
modelpath11 = "./models/fasterRCNNckpt"
destination11 = "./models/onnxmodels/fasterRCNN.onnx"
modelpath12 = "./models/pbmodels/linearRegression.pb"
modelpath12test = "./models/pbmodels"
destination12 = "./models/onnxmodels/linearRegression.onnx"
modelpath13 = "./models/googlenet"
destination13 = "./models/onnxmodels/googlenet.onnx"
modelpath14 = "./models/LeNet"
destination14 = "./models/onnxmodels/LeNet.onnx"
modelpath15 = "./models/unet"
destination15 = "./models/onnxmodels/unet.onnx"
modelpath16 = "./models/ssd"
destination16 = "./models/onnxmodels/ssd.onnx"
modelpath17 = "./models/pbmodels/ssd.pb"
destination17 = "./models/onnxmodels/ssd.onnx"
modelpath18 = "./models/models1_ckpt"
destination18 = "./models/onnxmodels/ckpt_linearregression.onnx"
modelpath19 = "./models/resnet18"
destination19 = "./models/onnxmodels/resnet18.onnx"
modelpath20 = "./models/vggnet16"
destination20 = "./models/onnxmodels/vggnet16.onnx"
modelpath21 = "./models/vggnet19"
destination21 = "./models/onnxmodels/vggnet19.onnx"

import tensorflow as tf
print(tf.__version__)
import os
import keras
import keras2onnx
import onnx

#从众多的文件中选择global_step最大的那个，即最新的那个
def find_latest(parent_path:str)->str:
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

def saved_as_pb(sess,savepath,outputname:list=None):
    """
    把加载的ckpt模型保存为pb的关键点在于固化节点的操作时要指定输出节点名称。
    因为网络其实是比较复杂的，定义了输出结点的名字，那么freeze的时候就只把输出该结点所需要的子图都固化下来，其他无关的就舍弃掉。因为我们freeze模型的目的是接下来做预测。所以，output_node_names一般是网络模型最后一层输出的节点名称，或者说就是我们预测的目标。
    """
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,outputname)
    with tf.gfile.FastGFile(savepath, mode='wb') as f:
        f.write(frozen_graph_def.SerializeToString())

#将savedmodel转为onnx不需要指定输入输出
def convert_savedmodel_into_onnx(modelpath,destinationpath,modelname=None):
    print(os.system("conda activate tensorflow-gpu && python -m tf2onnx.convert --saved-model %s --output %s"%(modelpath,destinationpath)))

convert_savedmodel_into_onnx(modelpath21,destination21)

def convert_h5_into_onnx(modelpath,destinationpath):
    # h5model = tf.keras.models.load_model(modelpath)#低版本的tensorflow(这里是1.11)继承的keras使用该句会出错KeyError: 'weighted_metrics'。这里换做keras自身，而不是继承的
    h5model = keras.models.load_model(modelpath)
    onnxmodel = keras2onnx.convert_keras(h5model,h5model.name)
    onnx.save_model(onnxmodel,destinationpath)

# convert_h5_into_onnx(modelpath5,destination5)
#


def convert_ckpt_into_onnx(modelpath,destinationpath,inputs:str=None,outputs:str=None):
    latest_meta_file = find_latest(modelpath)
    latest_ckpt = os.path.splitext(latest_meta_file)[0]
    print(latest_meta_file)
    print(os.system(
        "conda activate tensorflow-gpu && python -m tf2onnx.convert --checkpoint %s --inputs %s --outputs %s --output %s"%(
        latest_meta_file, inputs, outputs, destinationpath)))
    # with tf.Session() as sess:
    #     saver = tf.train.import_meta_graph(latest_meta_file)
    #     # 载入的是最新的那个ckpt文件，也就是数字最大的
    #     saver.restore(sess, latest_ckpt)
    #     graph = tf.get_default_graph()  # 获取当前图
    #     print([tensor.name for tensor in graph.as_graph_def().node])

    #     # saved_as_pb(sess,destinationpath,ckpt_outputs)


def convert_pb_into_onnx(modelpath,destinationpath,inputs=None,outputs=None):
    print(os.system("conda activate tensorflow-gpu && python -m tf2onnx.convert --input %s --inputs %s --outputs %s --output %s"%(modelpath,inputs,outputs,destinationpath)))

# convert_ckpt_into_onnx(modelpath16,destination16,"input/input_data","pred_sbbox/concat_2,pred_mbbox/concat_2,pred_lbbox/concat_2,pred_multi_scale/concat")
# convert_pb_into_onnx(modelpath12,destination12,"model_input:0","add:0")

# convert_savedmodel_into_onnx("./models/test_model","./models/onnxmodels/lenettest.onnx")