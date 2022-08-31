import keras
import keras2onnx
import onnx
import onnxruntime
from onnx2pytorch import ConvertModel
import torch
from onnx_tf.backend import prepare
onnx_model = onnx.load("./models/onnxmodel.onnx")
inference_model = onnxruntime.InferenceSession("./models/onnxmodel.onnx")
shape_without_dimension = inference_model.get_inputs()[0].shape[1:]
shape = [1]
shape.extend(shape_without_dimension)
print("输入数据形状:{}".format(shape))
pytorch_model = ConvertModel(onnx_model)
data = torch.randn(*shape)
predictions = pytorch_model(data).data.numpy()
print(predictions)
torch.save(pytorch_model,"./models/pytorchmodel.pth")