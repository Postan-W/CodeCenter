import onnx
import onnxruntime
import onnx2keras
model = onnx.load("./conv_classification2onnx.onnx")
runtimemodel = onnxruntime.InferenceSession("./conv_classification2onnx.onnx")
inputs = runtimemodel.get_inputs()[0].name
print(inputs)
keras_model = onnx2keras.onnx_to_keras(model,[inputs[:-2]])