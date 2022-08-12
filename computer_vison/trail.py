"""
@Time : 2021/5/26 11:33
@Author : wmingzhu
@Annotation : 
"""
modelpath = "C:\\Users\\15216\\Desktop\\models\\LeNet.onnx"
modelpath2 = "../model_transformation/onnxmodels/LeNetNoOpset.onnx"
import os
print(os.path.exists(modelpath2))
import onnx
import onnxruntime
from utility.model_process import universal_image_process
model = onnx.load(modelpath)
onnx.checker.check_model(model)
input = model.graph.input
#同理
output = model.graph.output
model2 = onnxruntime.InferenceSession(modelpath2)
inputs = model2.get_inputs()[0].name
print(type(inputs),inputs)
shape = model2.get_inputs()[0].shape[1:]
print(shape)
image = universal_image_process("../testimage.jpg",shape)
outputs = model2.get_outputs()[0].name
print(outputs)
result = model2.run([outputs],{inputs:image})
print(type(result))
print(result[0])


