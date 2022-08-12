import keras
import keras2onnx

h5model = keras.models.load_model("./models/2020.12.21-11.52.49keras_classification_1.h5")
print(h5model.name)
onnxmodel = keras2onnx.convert_keras(h5model,h5model.name,target_opset=9)
