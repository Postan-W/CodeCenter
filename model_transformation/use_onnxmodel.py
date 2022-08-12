"""
@Time : 2021/5/26 11:33
@Author : wmingzhu
@Annotation :
"""
modelpath = "C:\\Users\\15216\\Desktop\\models\\LeNet.onnx"
modelpath2 = "../model_transformation/onnxmodels/LeNetNoOpset.onnx"
modelpath3 = "./onnxmodels/case2_CNN.onnx"
modelpath4 = "./onnxmodels/miniVGGNet.onnx"
modelpath5 = "./onnxmodels/mobileNet_v1.onnx"
modelpath6 = "./onnxmodels/MobileNet_v2.onnx"
modelpath7 = "./models/onnxmodels/dcgan.onnx"
modelpath8 = "./models/onnxmodels/alexnet.onnx"
modelpath9 = "./models/onnxmodels/inception3.onnx"
modelpath10 = "./models/onnxmodels/inception4.onnx"
modelpath11 = "./models/onnxmodels/googlenet.onnx"
modelpath12 = "./models/onnxmodels/LeNet.onnx"
modelpath13 = "./models/onnxmodels/unet.onnx"
import os
print(os.path.exists(modelpath2))
import onnx
import onnxruntime
from utility.model_process import universal_image_process
#推理要用到onnxruntime库
model = onnx.load(modelpath)
onnx.checker.check_model(model)
input = model.graph.input
output = model.graph.output
model2 = onnxruntime.InferenceSession(modelpath13)
inputs = model2.get_inputs()[0].name
for input in model2.get_inputs():
    print(input.name)
print(type(inputs),inputs)
shape = model2.get_inputs()[0].shape[1:]
print(shape)
image = universal_image_process("./testimage.jpg",shape)
outputs = model2.get_outputs()[0].name
print(outputs)
result = model2.run([outputs],{inputs:image})
print(type(result))
print(result[0])


