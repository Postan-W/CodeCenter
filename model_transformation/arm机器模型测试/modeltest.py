import onnxruntime
from PIL import Image
import numpy as np
import logging
from flask import Flask,request,make_response
app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s || information=%(message)s',datefmt="%y-%m-%d %H:%M:%S")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)
@app.route("/predict/<int:number>",methods=["GET"])
def predict(number):
    model = onnxruntime.InferenceSession("./models/test3.onnx")
    inputs = model.get_inputs()[0].name
    outputs = model.get_outputs()[0].name
    shape = model.get_inputs()[0].shape[1:]
    print(inputs,outputs,shape)
    with_batch = [1]
    with_batch.extend(shape)
    image = Image.open("./pictures/"+str(number)+".jpg")
    image = image.resize((shape[1], shape[0]))
    image = image.convert("RGB")
    print(image.size)
    image_numpy = np.array(image).reshape(with_batch).astype("float32")
    result = model.run([outputs], input_feed={inputs: image_numpy})
    logger.info("预测结果是:"+str(result[0]))
    return make_response("预测结果是:"+str(result[0]),200)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
