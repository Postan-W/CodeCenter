import tensorflow as tf
import os
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
model_dir = "./models1_ckpt/linearregression.ckpt-9"#这里载入最新的

print_tensors_in_checkpoint_file(model_dir,None,True)
print_tensors_in_checkpoint_file(model_dir,tensor_name='add:0',all_tensors=False)

def print_all_nodes(graph):
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]
    print(tensor_name_list)

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
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #一个协议缓冲，保存tensorflow中完整的graph、variables、operation、collection；这是我们恢复模型结构的参照
    meta_file = find_latest("./models1_ckpt")
    latest_ckpt = os.path.splitext(meta_file)[0]#也可以用tf.train.latest_checkpoint("./models1_ckpt")来获得。
    print(latest_ckpt)
    latest_ckpt = tf.train.latest_checkpoint("./models1_ckpt")
    print(latest_ckpt)
    saver = tf.train.import_meta_graph(meta_file)
    #载入的是最新的那个ckpt文件，也就是数字最大的
    saver.restore(sess,latest_ckpt)

    graph =tf.get_default_graph()#获取当前图,也就是import_meta_graph恢复的图
    #上面获取的图等价于sess.graph获取的图
    #graph = sess.graph
    #打印所有节点的名称
    print([tensor.name for tensor in graph.as_graph_def().node])

    print(tf.trainable_variables())

    x = graph.get_tensor_by_name("model_input:0")
    z = graph.get_tensor_by_name("add:0")
    print(sess.run(z, feed_dict={x: 20}))