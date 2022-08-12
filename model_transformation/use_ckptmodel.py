"""
@Time : 2021/6/4 17:48
@Author : wmingzhu
@Annotation : 
"""
import tensorflow as tf
import os


modelpath = "./models/fasterRCNNckpt"
modelpath2 = "./models/ssd"
"""
要得到的是output.model-3.index、output.model-3.meta、output.model-3.data-00000-of-00001前面的部分output.model-3，这是要restore函数要的参数
"""
#从众多的文件中选择global_step最大的那个，即最新的那个
def find_latest(parent_path:str)->str:
    file_list = os.listdir(parent_path)
    count = 0
    max = 0
    meta_file = ""
    for file in file_list:
        if file.endswith(".meta"):
            count += 1

    if count == 1:
        for file in file_list:
            if file.endswith(".meta"):
                meta_file = parent_path + file if parent_path.endswith("/") else parent_path +"/" + file
                break
    else:
        for file in file_list:
            if file.endswith(".meta"):
                number = int(os.path.splitext(file)[0][-1])
                max = number if max < number else max

        for file in file_list:
            if file.endswith(".meta"):
                if int(os.path.splitext(file)[0][-1]) == max:
                    meta_file = parent_path + file if parent_path.endswith("/") else parent_path + "/" + file
                    break

    return meta_file

def saved_as_pb(sess,savepath,outputname:list=None):
    """
    sess: 当前使用的会话对象sess
    input_graph_def: 是一个GraphDef object ，及当前会话中的Graph
    output_node_names: graph输出节点的名称，形如 [“name1”，“name2” ]
    variable_names_whitelist: 需要转化的变量Variable所组成的list，默认情况下graph中的所有     variable均会转化成constant(by default, all variables are converted).
    variable_names_blacklist: 忽略转化。即不需要转化成constant的variables所组成的list
    """
    #var_list = tf.trainable_variables()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,outputname)
    with tf.gfile.FastGFile(savepath, mode='wb') as f:
        f.write(frozen_graph_def.SerializeToString())

def use_ckpt():
    latest_meta_file = find_latest(modelpath2)
    # 根据metafile文件来找到最新的ckpt
    latest_ckpt = os.path.splitext(latest_meta_file)[0]
    latest_ckpt = tf.train.latest_checkpoint(modelpath2)
    print(latest_ckpt)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(latest_meta_file)
        # 载入的是最新的那个ckpt文件，也就是数字最大的
        saver.restore(sess, latest_ckpt)

        graph = tf.get_default_graph()  # 获取当前图
        print([tensor.name for tensor in graph.as_graph_def().node])
        # input = graph.get_tensor_by_name()
        # output = graph.get_tensor_by_name()
        saved_as_pb(sess,"./models/pbmodels/ssd.pb",['global_step'])

use_ckpt()
#导出静态图，通过tensorboard查看
from tensorflow.summary import FileWriter
def get_graph():
    with tf.Session() as sess:
        latest_meta = tf.train.latest_checkpoint(modelpath2)+".meta"
        tf.train.import_meta_graph(latest_meta)
        FileWriter("./tensorboard",sess.graph)


# get_graph()






