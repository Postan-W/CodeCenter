import torch
import torchvision.models as buildinmodels
from onnx_tf.backend import prepare
import onnx
from tensorflow.python import keras
import onnxruntime
import numpy as np
import tensorflow.compat.v1 as tf
import os
import keras2onnx
from onnx2pytorch import ConvertModel
import onnx2keras
import tf2onnx
import torch.nn.functional as functional
from PIL import Image
from pytorch2keras import pytorch_to_keras,converter
from trans_utility import h5_input_shape
from trans_utility import logger,remove_model,dir_dict,model_dict,copyfiles,remove_temp_savedmodel
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
"""
Tensorflow: BHWC;Pytorch: BCHW;尽管不同框架处理的数据维度顺序不同，但转换的模型始终保持同一种数据格式输入，不会影响结果，
这说明处理原始模型foward过程在转换为其他框架后对数据的维度处理仍然保持原始框架的顺序
"""

class ModelTrans:
    def __init__(self,destination):
        self.destination = destination
        self.inputname = "inputtest"
        self.outputname = "outputtest"
        self.shape = [1,28,28]#默认CHW
        self.__get_frame_name()
        logger.info("将{}转为{}".format(self.source, self.destination))

    def __get_frame_name(self):
        try:
            model = os.listdir(dir_dict[1])[0]
        except Exception as e:
            logger.info("{}:源模型不存在".format(e))
        file_type = model[model.rfind(".")+1:]
        frame_name = None
        for key in model_dict.keys():
            if file_type in model_dict[key]:
                frame_name = key
                break
        self.source = frame_name

    @staticmethod
    def pb2onnx(inputs,outputs):
        model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
        remove_model(2)
        file = os.listdir(dir_dict[1])[0]
        filename = file[:file.rfind(".")]
        target_name = filename + "2onnx.onnx"
        #需要先把pb转为savedmodel
        remove_temp_savedmodel()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            with open(model_path, 'rb') as f:
                graph_def = tf.GraphDef()  # 序列化的图对象，可以接收序列化数据形成还原图的信息
                graph_def.ParseFromString(f.read())
                tf.get_default_graph()
                inputtensor, outputtensor = tf.import_graph_def(graph_def,
                                                                return_elements=[inputs+":0", outputs+":0"])
                builder = tf.saved_model.builder.SavedModelBuilder("./temp_savedmodel")
                signature = tf.saved_model.signature_def_utils.predict_signature_def(inputs={inputs: inputtensor},
                                                                                     outputs={
                                                                                         outputs: outputtensor})
                builder.add_meta_graph_and_variables(sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
                                                     signature_def_map={"test_signature": signature})
                builder.save()
                print("pb到savedmodel转换成功")

        logger.info("转换返回状态(0代表正常):", os.system(
            "python -m tf2onnx.convert --saved-model %s  --output %s" %(
             "./temp_savedmodel", dir_dict[2]+target_name)))

    @staticmethod
    def h52onnx():
        try:
            model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
            h5model = keras.models.load_model(model_path)
            logger.info("加载h5模型成功")
            onnxmodel = keras2onnx.convert_keras(h5model, h5model.name)
            remove_model(2)
            file = os.listdir(dir_dict[1])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2onnx.onnx"
            onnx.save_model(onnxmodel, dir_dict[2]+target_name)
        except Exception as e:
            logger.info("h5_to_onnx出错:{}".format(e))

    @staticmethod
    def pytorch2onnxfunction(shape:list, inputname:str, outputname:str):
        try:
            model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
            model = torch.load(model_path)
            logger.info("pytorch模型加载完成")
            shape_with_batch = [2]
            shape_with_batch.extend(shape)
            x = torch.randn(*shape_with_batch)#样例数据
            remove_model(2)
            file = os.listdir(dir_dict[1])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2onnx.onnx"
            torch.onnx.export(model, x, dir_dict[2]+target_name, input_names=[inputname], output_names=[inputname],
                              dynamic_axes=
                              {inputname: {0: "batch_size"}, outputname: {0: "batch_size"}})
        except Exception as e:
            logger.info(e)
    @staticmethod
    def pytorch2kerash5(shape:list):
        try:
            model_path = os.path.join(dir_dict[1], os.listdir(dir_dict[1])[0])
            torch_model = torch.load(model_path)
            logger.info("加载Pytorch模型成功")
            input_shape = [1]  # 第一维是batch
            input_shape.extend(shape)
            logger.info("Pytorch转Keras，输入形状为:{}".format(input_shape))
            keras_model = converter.pytorch_to_keras(model=torch_model, change_ordering=True,args=torch.autograd.Variable(
                torch.FloatTensor(np.random.uniform(0, 1, input_shape))), input_shapes=[shape],
                                          verbose=True)
            logger.info("Pytorch2Keras成功")
            remove_model(3)
            file = os.listdir(dir_dict[1])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2h5.h5"
            keras.models.save_model(keras_model, dir_dict[3] + target_name)
        except Exception as e:
            logger.info("Pytorch2Keras失败:{}".format(e))

    @staticmethod
    def onnx2pb():
        if len(os.listdir(dir_dict[2])) != 0:
            remove_model(3)
            file = os.listdir(dir_dict[2])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2pb.pb"
            try:
                onnx_model = onnx.load(os.path.join(dir_dict[2], os.listdir(dir_dict[2])[0]))
                logger.info("加载onnx成功")
            except Exception as e:
                logger.info("加载onnx失败:{}".format(e))
            try:
                tf_exp = prepare(onnx_model)  # prepare tf representation
                tf_exp.export_graph(dir_dict[3] + target_name)
                logger.info("onnx到pb成功")
            except Exception as e:
                logger.info("onnx到pb失败:{}".format(e))
                copyfiles(2,3)

    @staticmethod
    def onnx2h5(inputname:str,inputshape:list=None):
        if len(os.listdir(dir_dict[2])) != 0:
            remove_model(3)
            file = os.listdir(dir_dict[2])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2h5.h5"
            try:
                onnx_model = onnx.load(os.path.join(dir_dict[2], os.listdir(dir_dict[2])[0]))
                logger.info("加载onnx成功")
            except Exception as e:
                logger.info("加载onnx失败:{}".format(e))
            try:
                keras_model = onnx2keras.onnx_to_keras(onnx_model,[inputname])
                keras.models.save_model(keras_model, dir_dict[3]+target_name, include_optimizer=True)
                logger.info("h5模型生成成功")
            except Exception as e:
                logger.info("onnx到h5失败:{}".format(e))
                copyfiles(2, 3)

    @staticmethod
    def onnx2pytorch():
        try:
            onnx_model_path = os.path.join(dir_dict[2], os.listdir(dir_dict[2])[0])
            onnx_model = onnx.load(onnx_model_path)
            inference_model = onnxruntime.InferenceSession(onnx_model_path)#推理
            logger.info("onnx模型加载成功")
            shape_without_dimension = inference_model.get_inputs()[0].shape[1:]
            shape = [3]
            shape.extend(shape_without_dimension)
            logger.info("输入数据形状:{}".format(shape))
            pytorch_model = ConvertModel(onnx_model)
            logger.info("onnx2pytorch转换成功")
            data = torch.randn(*shape)
            predictions = pytorch_model(data).data.numpy()
            logger.info("用onnx2pytorch得到的模型预测:{}".format(predictions))
            remove_model(3)
            file = os.listdir(dir_dict[2])[0]
            filename = file[:file.rfind(".")]
            target_name = filename + "2pth.pth"
            torch.save(pytorch_model, dir_dict[3]+target_name)
        except Exception as e:
            logger.info("onnx2pytorch出错:{}".format(e))

    def transformer(self):
        if self.source == "Pytorch":
            if self.destination == "Tensorflow":
                ModelTrans.pytorch2onnxfunction(self.shape, self.inputname, self.outputname)
                ModelTrans.onnx2pb()
            if self.destination == "Keras":
                ModelTrans.pytorch2kerash5(self.shape)
        elif self.source == "Keras":
            if self.destination == "Pytorch":
                ModelTrans.h52onnx()
                ModelTrans.onnx2pytorch()
            if self.destination == "Tensorflow":
                ModelTrans.h52onnx()
                ModelTrans.onnx2pb()
        elif self.source == "Tensorflow":
            if self.destination == "Pytorch":
                ModelTrans.pb2onnx(self.inputname,self.outputname)
                ModelTrans.onnx2pytorch()
            if self.destination == "Keras":
                ModelTrans.pb2onnx(self.inputname,self.outputname)
                ModelTrans.onnx2h5(self.inputname)

    def h5_predictor(self):
        try:
            model = keras.models.load_model(os.path.join(dir_dict[3]+os.listdir(dir_dict[3])[0]))
            model_input = model.input
            model_output = model.output
            shape_withoutbatch = h5_input_shape(model.to_json())
            logger.info("模型输入的形状是{}".format(shape_withoutbatch))
            image = Image.open(os.path.join("./image_for_predict",os.listdir("./image_for_predict")[0]))
            logger.info("图片的尺寸是:{}".format(image.size))
            try:
                image = image.resize((shape_withoutbatch[1],shape_withoutbatch[0]))
                with_batch = [1]
                with_batch.extend(shape_withoutbatch)
                image_numpy = np.array(image).reshape(with_batch)
                predictions = model.predict(image_numpy)
                return predictions
            except:
                logger.info("》》》》》》》》通道在后面《《《《《《《《")
                image.resize((shape_withoutbatch[1], shape_withoutbatch[0]))
                with_batch = [1]
                with_batch.extend(shape_withoutbatch)
                image_numpy = np.array(image).reshape(with_batch)
                predictions = model.predict(image_numpy)
                return predictions
        except Exception as e:
            return e

    def pb_predictor(self):
        with tf.Graph().as_default():
            output_graph_def = tf.GraphDef()
            output_graph_path = dir_dict[3]+os.listdir(dir_dict[3])[0]
            with open(output_graph_path, 'rb') as f:
                output_graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(output_graph_def, name="")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                try:
                    input = sess.graph.get_tensor_by_name(self.inputname + ".1:0")
                except:
                    logger.info("使用传入的输入输出")
                    input = sess.graph.get_tensor_by_name(self.inputname + ":0")
                input_shape = list(input.shape)[1:]
                try:
                    output = sess.graph.get_tensor_by_name(self.inputname + ":0")
                except:
                    output = sess.graph.get_tensor_by_name(self.outputname + ":0")

                image = Image.open(os.path.join("./image_for_predict", os.listdir("./image_for_predict")[0]))
                logger.info("图片的尺寸是:{}".format(image.size))
                try:
                    image.resize((input_shape[2], input_shape[1]))
                    with_batch = [1]
                    with_batch.extend(input_shape)
                    image_numpy = np.array(image).reshape(with_batch)
                    predictions = sess.run(output, feed_dict={input:image_numpy})
                except:
                    image.resize((input_shape[1], input_shape[0]))
                    with_batch = [1]
                    with_batch.extend(input_shape)
                    image_numpy = np.array(image).reshape(with_batch)
                    predictions = sess.run(output, feed_dict={input: image_numpy})
                return predictions

    def pytorch_predictor(self):
        model = torch.load(dir_dict[3]+os.listdir(dir_dict[3])[0])
        model.eval()
        image = Image.open(os.path.join("./image_for_predict", os.listdir("./image_for_predict")[0]))
        image.resize((self.shape[2], self.shape[1]))
        with_batch = [1]
        with_batch.extend(self.shape)
        image_numpy = np.array(image).reshape(with_batch)
        predictions = model(torch.Tensor(image_numpy))
        return predictions

    def onnx_preditor(self):
        model = onnxruntime.InferenceSession(dir_dict[3] + os.listdir(dir_dict[3])[0])
        inputs = model.get_inputs()[0].name
        outputs = model.get_outputs()[0].name
        shape = model.get_inputs()[0].shape[1:]
        with_batch = [1]
        with_batch.extend(shape)
        image = Image.open(os.path.join("./image_for_predict", os.listdir("./image_for_predict")[0]))
        image = image.resize((shape[1], shape[0]))
        image_numpy = np.array(image).reshape(with_batch).astype("float32")
        result = model.run([outputs], input_feed={inputs: image_numpy})
        return result[0]

    #仅针对图片作预测
    def predict(self):
        if os.listdir(dir_dict[3])[0].endswith("onnx"):
            logger.info("onnx转为目标模型失败，现使用Onnx模型作预测")
            return self.onnx_preditor()
        elif self.destination == "Keras":
            return self.h5_predictor()
        elif self.destination == "Tensorflow":
                model = onnxruntime.InferenceSession(dir_dict[2] + os.listdir(dir_dict[2])[0])
                inputs = model.get_inputs()[0].name
                outputs = model.get_outputs()[0].name
                shape = model.get_inputs()[0].shape[1:]
                with_batch = [1]
                with_batch.extend(shape)
                image = Image.open(os.path.join("./image_for_predict", os.listdir("./image_for_predict")[0]))
                try:
                    image1 = image.resize((shape[1], shape[0]))
                    image_numpy = np.array(image1).reshape(with_batch).astype("float32")
                except:
                    image2 = image.resize((shape[2], shape[1]))
                    image_numpy = np.array(image2).reshape(with_batch).astype("float32")
                result = model.run([outputs], input_feed={inputs: image_numpy})[0]
                return result
        elif self.destination == "Pytorch":
            return self.pytorch_predictor()



def test():
    pth_model = torch.load("./models/mnist_classification_epoch2.pth")
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    p0, p2, p8, p9, p4, p6 = x_train[1], x_train[12000], x_train[24000], x_train[36000], x_train[48000], x_train[59000]
    images = np.array(
        [p0.reshape((1, 28, 28)), p2.reshape((1, 28, 28)), p8.reshape((1, 28, 28)), p9.reshape((1, 28, 28)),
         p4.reshape((1, 28, 28)), p6.reshape((1, 28, 28))])
    print(images.shape)
    pth_predictions = pth_model(torch.Tensor(images))
    print("pth的预测结果:",pth_predictions.detach().numpy().argmax(axis=1))
    onnx_model = onnxruntime.InferenceSession("./intermediate_models/mnist_classification_epoch2_pth2onnx.onnx")
    onnx_predictions = onnx_model.run(["outputtest"], {"inputtest": images.astype("float32")})
    print("onnx预测结果:", np.array(onnx_predictions[0]).argmax(axis=1))
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        output_graph_path = "./transformed_models/mnist_classification_epoch2_pth2onnx2pb.pb"
        with open(output_graph_path, 'rb') as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.graph.get_tensor_by_name("outputtest:0")
            input = sess.graph.get_tensor_by_name("inputtest:0")
            predictions = sess.run(output, feed_dict={input: images.astype("float32")})
            print("pb预测结果:", np.array(predictions).argmax(axis=1))
