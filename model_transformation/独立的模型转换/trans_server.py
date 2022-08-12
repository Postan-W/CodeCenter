import json
from flask import Flask, jsonify, request,make_response,send_from_directory
import requests
import os
from flask_cors import CORS
from self_build import MnistClassificationDynamicInput
from transformer_and_predictor import ModelTrans
from trans_utility import dir_dict,logger,remove_model
app = Flask(__name__)
CORS(app, supports_credentials=True)#不跨域的话前后端在一台机器上没法互访
trans = None
# trans = ModelTrans("Pytorch")
# trans.shape = [28,28,1]
# trans.pb2onnx("model_input:0","add:0")
#上传、转换、预测、下载的结果共用一个前端展示位
@app.route("/upload_model",methods=["POST"])
def upload_model():
    try:
        model_file = request.files["model"]
        remove_model(1)
        logger.info("上传的模型是:{}".format(model_file.filename))
        model_file.save(dir_dict[1] + model_file.filename)
        return "200 模型上传成功"
    except Exception as e:
        return "模型上传失败:"+str(e)

@app.route("/transform",methods=["POST"])
def transform():
    if len(os.listdir(dir_dict[1])) == 0:
        return "请先上传模型"
    else:
        model_information = json.loads(request.data)
        logger.info("接收的数据是:",model_information)
        target_framework = model_information["targetFramework"]
        input_shape = model_information["inputShape"]
        logger.info(type(input_shape))
        input_name = model_information["inputName"]
        output_name = model_information["outputName"]
        global trans
        trans = ModelTrans(target_framework)
        trans.inputname,trans.outputname,trans.shape = input_name,output_name,input_shape
        print(trans.inputname, trans.outputname, trans.shape)
        trans.transformer()
        return "200 转换成功，可以下载或预测"

@app.route("/download_model/",methods=["GET"])
def download_model():
    try:
        print(trans.inputname, trans.outputname, trans.shape, trans.destination)
        logger.info("下载的模型为:{}".format(os.listdir("./transformed_models")[0]))
        return send_from_directory("./transformed_models", filename=os.listdir("./transformed_models")[0],
                                   as_attachment=True)
    except Exception as e:
        logger.info("没有转换成功的模型可供下载")
        return str(e)

#接收上传的一张图片保存到image文件夹下，然后用该图片预测
@app.route("/predict",methods=["POST"])
def predict():
    if trans == None:
        return "404 请先转换模型，然后进行预测"
    else:
        if len(os.listdir("./image_for_predict")) != 0:
            os.remove(os.path.join("./image_for_predict/",os.listdir("./image_for_predict")[0]))
        new_image = request.files["image"]
        new_image.save("./image_for_predict/"+new_image.filename)
        return {"result":str(trans.predict())}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
