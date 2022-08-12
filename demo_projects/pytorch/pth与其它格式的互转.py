#Pytorch中tensor维度顺序为B,C,H,W
import torch
from onnx_tf.backend import prepare
import onnx
import keras
import onnxruntime
import numpy as np
import tensorflow.compat.v1 as tf
import os
import keras2onnx
from onnx2pytorch import ConvertModel
import onnx2keras
import tf2onnx
import torch.nn.functional as functional
from utility.model_process import h5_input_shape
from pytorch2keras import pytorch_to_keras
from PIL import Image

#Pytorch加载自定义模型时，由于pickle的操作，自定义模型类需要能在当前模块找到，要么导入要么在当前模块给出
class MnistClassificationDynamicInput(torch.nn.Module):
    def __init__(self,h,w):
        super(MnistClassificationDynamicInput,self).__init__()
        h1 = int((h-5+1)/2)
        w1 = int((w-5+1)/2)
        h2 = int((h1-4)/2)
        w2 = int((w1-4)/2)
        self.convolution1 = torch.nn.Conv2d(3,10,kernel_size=5)#输入单通道，输出10通道，卷积核大小(5,5)
        self.convolution2 = torch.nn.Conv2d(10,20,kernel_size=5)#输入10通道，输出20通道，卷积核大小(5,5)
        self.convolution2_drop = torch.nn.Dropout2d()
        self.k = 20*h2*w2
        print("全连接层的输入大小为:",self.k)
        self.full_connection1 = torch.nn.Linear(self.k,60)
        self.full_connection2 = torch.nn.Linear(60,10)

    def forward(self,x):
        x = functional.relu(functional.max_pool2d(self.convolution1(x),2))#此时尺寸为(10,12,12)
        x = functional.relu(functional.max_pool2d(self.convolution2_drop(self.convolution2(x)),2))#此时尺寸为(20,4,4)，所以共320个神经元
        x = x.view(-1,self.k)#展平操作
        x = functional.relu(self.full_connection1(x))
        x = functional.dropout(x,training=self.training)#training=True,训练的时候才dropout
        x = self.full_connection2(x)
        return functional.softmax(x)


# torch_model = torch.load("./models/flowers_class.pth")
# model = pytorch_to_keras(model=torch_model,args=torch.autograd.Variable(torch.FloatTensor(np.random.uniform(0, 1, (1, 3, 320, 320)))),input_shapes=[(3,320,320)],change_ordering=True,verbose=True)
# keras.models.save_model(model,"./models/flowers_class_pth.h5")

def use_pth(modelpath,inputshape):
    model = torch.load(modelpath)
    with_batch = [1]
    with_batch.extend(inputshape)
    image_numpy = np.random.randn(*with_batch).astype("float32")
    predictions = model(torch.Tensor(image_numpy))
    print(predictions)

# use_pth("./models/flowers_class.pth",[3,320,320])

def pth_to_onnx(sourcepath,destinationpath,shape):
    print("pth====>>onnx")
    model = torch.load(sourcepath)
    print("模型加载完成")
    # print(linear_model.eval())
    # model = buildinmodels.resnet18()
    # print(resnet18.eval())
    x = torch.randn(*shape)
    # resnet18_predictions = model(x)
    torch.onnx.export(model,x, destinationpath,input_names=["inputtest"],output_names=["outputtest"],dynamic_axes=
    {"inputtest": {0: "batch_size"},"outputtest": {0: "batch_size"}})#必须要说明首个维度为batch维,否则第一个维度的数值就被固定了
    #可视化网络可以看到这里设定的输入输出名称即便在转为其它格式的模型后，其它模型依然采用该名称
# pth_to_onnx('./models/mnist_classification_softmax_epoch10.pth',"./models/mnist_classification_softmax_epoch10_pth2onnx.onnx",[3,1,28,28])

def load_onnx(path):
    print("加载onnx模型")
    onnx_model = onnx.load(path)
    inference_model = onnxruntime.InferenceSession(path)
    onnx.checker.check_model(onnx_model)
    inuput_tensor = onnx_model.graph.input
    print(inuput_tensor)
    output_tensor = onnx_model.graph.output
    print(output_tensor)
    # print(onnx.helper.printable_graph(onnx_model.graph))
    input = inference_model.get_inputs()[0].name
    output = inference_model.get_outputs()[0].name
    print(input,output)
    print("get_inputs:",inference_model.get_inputs()[0])
    shape_without_dimension = inference_model.get_inputs()[0].shape[1:]
    image = Image.open("./mnist_based/8.jpg")
    image.resize((shape_without_dimension[2], shape_without_dimension[1]))
    with_batch = [1]
    with_batch.extend(shape_without_dimension)
    image_numpy = np.array(image).reshape(with_batch).astype("float32")
    print(image_numpy.shape)
    # shape = [3]
    # shape.extend(shape_without_dimension)
    # print(shape)
    # x = torch.randn(*shape)
    # print("测试数据:{}".format(np.array(x).dtype))
    result = inference_model.run([output],{input:image_numpy})
    print(result[0])
# load_onnx("./models/cnn-functional.onnx")

def onnx_to_pb(sourcepath,destinationpath):
    print("onnx====>>pb")
    onnx_model = onnx.load(sourcepath)
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(destinationpath)

# onnx_to_pb("./models/cnn-functional.onnx","./models/cnn-functional.pb")

def use_pbmodel(path,input,output):#默认一个输入输出
    print("加载pb模型")
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = path
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            input = sess.graph.get_tensor_by_name(input)
            output = sess.graph.get_tensor_by_name(output)
            shape = [3]
            shape.extend(list(input.shape)[1:])
            print(shape)
            input_data = np.random.randint(1,10,shape,'int32')# 输入要测试的数据，格式要一致
            # print("测试数据:{}".format(input_data))
            predictions = sess.run(output, feed_dict={input: input_data})
            print("predictions:", predictions)
# use_pbmodel("./models/cnn-functional.pb","input_1:0","dense:0")

def  onnx_to_h5(sourcepath,destinationpath):
    print("onnx====>>h5")
    onnx_model = onnx.load(sourcepath)
    keras_model = onnx2keras.onnx_to_keras(onnx_model,["inputtest"])
    #用tf2集成的Keras，以免版本不符导致错误
    keras.models.save_model(keras_model,destinationpath,include_optimizer=True)
# onnx_to_h5("./models/mnist_classification_softmax_epoch10_pth2onnx.onnx","./models/mnist_classification_softmax_epoch10_pth2onnx2h5.h5")

def use_h5model(path):
    print("加载h5模型")
    model = keras.models.load_model(path)
    inputname = model.input
    outputname = model.output
    # keras.backend.set_image_data_format('channels_first')#该句没起作用
    shape_withoutdimension = h5_input_shape(model.to_json())
    shape = [1]
    shape.extend(shape_withoutdimension)
    print("输入数据形状:{}".format(shape))

    image_numpy = np.random.randn(*shape).astype("float32")
    prediction = model.predict(image_numpy)
    print(np.sum(prediction,axis=1))
    print(np.max(prediction))
    print("预测结果:{}".format(prediction))
# use_h5model("./models/flowers_class_pth.h5")

def onnx_to_pth(sourcepath,destinationpath):
    print("onnx====>>pth")
    onnx_model = onnx.load(sourcepath)
    inference_model = onnxruntime.InferenceSession(sourcepath)
    shape_without_dimension = inference_model.get_inputs()[0].shape[1:]
    shape = [3]
    shape.extend(shape_without_dimension)
    print("输入数据形状:{}".format(shape))
    pytorch_model = ConvertModel(onnx_model)
    data = torch.randn(*shape)
    predictions = pytorch_model(data).data.numpy()
    print(predictions)
    print(np.sum(predictions,axis=1))
    # result = np.array(pytorch_model(data))
    # print("预测结果:{}".format(result))
    torch.save(pytorch_model,destinationpath)

# onnx_to_pth("./models/cnn-functional.onnx","./models/cnn-functional.pth")

def convert_h5_into_onnx(modelpath,destinationpath):
    # h5model = tf.keras.models.load_model(modelpath)#低版本的tensorflow(这里是1.11)集成的keras使用该句会出错KeyError: 'weighted_metrics'。这里换做keras自身，而不是继承的
    print("h5====>>onnx")
    h5model = keras.models.load_model(modelpath)
    print("h5模型名称:",h5model.name)
    onnxmodel = keras2onnx.convert_keras(h5model,h5model.name)
    onnx.save_model(onnxmodel,destinationpath)

convert_h5_into_onnx("./models/2021.01.15-11.40.10keras_classification_2.h5","./models/mnistclissification.onnx")

def convert_h5_to_savedmodel(sourcepath,destinationpath):
    print("h5====>>savedmodel")
    h5model = keras.models.load_model(sourcepath)
    tf.saved_model.save(h5model,destinationpath)

# convert_h5_to_savedmodel("resnet18.h5","resnet18h5_to_savedmodel")
def convert_saved_model_to_onnx(sourcepath,destinationpath):
    print("savedmodel====>>onnx")
    print(os.system("conda activate tf2.0 && python -m tf2onnx.convert --saved-model %s --output %s" % (
    sourcepath, destinationpath)))

# convert_saved_model_to_onnx("resnet18h5_to_savedmodel","resnet18savedmodel_to_onnx.onnx")



