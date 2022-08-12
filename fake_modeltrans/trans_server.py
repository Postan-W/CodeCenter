from flask import Flask,request,make_response,send_from_directory
import os
import random
import json
import datetime
import time as tm
import onnxruntime
from flask_cors import CORS
from PIL import Image
import numpy as np
import logging
app = Flask(__name__)
CORS(app, supports_credentials=True)
target_framework = None
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s || information=%(message)s',datefmt="%y-%m-%d %H:%M:%S")
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(console)
@app.route("/upload_model",methods=["POST"])
def upload_model():
        try:
            entire_filepath = os.path.join("./uploadmodel", os.listdir("./uploadmodel")[0])
            os.remove(entire_filepath)
            print("已删除前一个上传的模型")
        except:
            print("第一次上传，不需要删除模型")
        model_file = request.files["model"]
        model_file.save("./uploadmodel/"+ model_file.filename)
        return make_response("模型上传成功",200)
        return make_response("上传失败{}".format(e),500)

@app.route("/transform",methods=["POST"])
def transform():
    #假转换，随机模型转换时长
    global target_framework
    model_information = json.loads(request.data)
    target_framework = model_information["targetFramework"]
    time = random.randint(3,8)
    starttime = datetime.datetime.now()
    tm.sleep(time)
    endtime = datetime.datetime.now()
    print("转换耗时:{}".format(endtime-starttime))
    return make_response("模型转换成功", 200)

@app.route("/download_model",methods=["GET"])
def download_model():
    try:
        filename = None
        models = os.listdir("./downloadmodels")
        for model in models:
            if model.startswith(target_framework):
                filename = model
                break
        logger.info(filename)
        response = make_response(send_from_directory("./downloadmodels", filename=filename,
                                   as_attachment=True))
        response.headers["Content-Disposition"] = "attachment; filename={}".format(filename.encode().decode('latin-1'))
        return response 
    except Exception as e:
        logger.info(e)
        return make_response("下载失败", 500)

#接收上传的一张图片保存到image文件夹下，然后用该图片预测
@app.route("/predict",methods=["POST"])
def predict():
    new_image = request.files["image"]
    try:
        entire_filepath = os.path.join("./image_for_predict/", os.listdir("./image_for_predict/")[0])
        os.remove(entire_filepath)
    except:
        print("第一次预测，不需要删除图片")
    new_image.save("./image_for_predict/" + new_image.filename)
    model = onnxruntime.InferenceSession("./predictmodel/flowers_cli_dense2onnx.onnx")
    inputs = model.get_inputs()[0].name
    outputs = model.get_outputs()[0].name
    shape = model.get_inputs()[0].shape[1:]
    with_batch = [1]
    with_batch.extend(shape)
    image = Image.open("./image_for_predict/" + new_image.filename)
    image = image.resize((shape[1], shape[0]))
    image_numpy = np.array(image).reshape(with_batch).astype("float32")
    result = model.run([outputs], input_feed={inputs: image_numpy})
    return make_response("预测结果是:"+str(result[0]),"200")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
