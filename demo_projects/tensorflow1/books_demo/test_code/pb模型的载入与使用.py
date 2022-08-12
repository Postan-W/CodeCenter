import tensorflow as tf
from google.protobuf import text_format
model_path = "../test_code/models3_pb/linearRegression.pb"
model_path2 = "../test_code/models3_pb/linearRegression2.pb"
def load_and_predict():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.gfile.FastGFile(model_path2,'rb',) as f:
            graph_def = tf.GraphDef()#序列化的图对象，可以接收序列化数据形成还原图的信息
            print(graph_def)
            # graph_def.ParseFromString(f.read())
            print("----------------------------------------------------------------------")
            text_format.Merge(f.read(), graph_def)
            print(graph_def)

            # print(tf.get_default_graph())
            #sess.graph.as_default()
            input_tensor,output_tensor = tf.import_graph_def(graph_def,return_elements=["model_input:0","add:0"],name="")#名字为空
            print("----------------------------------------------------------------------")
            print(sess.graph_def)
            #上面一行代码可以用下面两行代替,其中sess.graph和tf.get_default_graph()返回的都是同一个graph
            # input_tensor = sess.graph.get_tensor_by_name("model_input:0")
            # output_tensor = sess.graph.get_tensor_by_name("add:0")
            print("输入形状:",tf.shape(input_tensor))
            print("输出形状：",tf.shape(output_tensor))
            print(sess.run(output_tensor,feed_dict={input_tensor:20}))

def load_and_predict2():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with tf.gfile.FastGFile(model_path2,'rb') as f:
            print(sess.graph)
            graph_def = tf.GraphDef()#序列化的图对象，可以接收序列化数据形成还原图的信息
            graph_def.ParseFromString(f.read())
            print(sess.graph)
            print(tf.get_default_graph())
            #sess.graph.as_default()
            input_tensor,output_tensor = tf.import_graph_def(graph_def,return_elements=["model_input:0","add:0"],name="")#名字为空
            #上面一行代码可以用下面两行代替,其中sess.graph和tf.get_default_graph()返回的都是同一个graph
            # input_tensor = sess.graph.get_tensor_by_name("model_input:0")
            # output_tensor = sess.graph.get_tensor_by_name("add:0")
            print("输入形状:",tf.shape(input_tensor))
            print("输出形状：",tf.shape(output_tensor))
            print(sess.run(output_tensor,feed_dict={input_tensor:20}))

class PbModel():
    sess = tf.Session()
    def __init__(self,model_path):
        f = tf.gfile.FastGFile(model_path,'rb')
        self.graph_def = tf.GraphDef()
        self.graph_def.ParseFromString(f.read())

    def predict(self,data):
        input_tensor,output_tensor = tf.import_graph_def(self.graph_def,return_elements=["model_input:0","add:0"],name="")

        return PbModel.sess.run(output_tensor,feed_dict={input_tensor:data})

load_and_predict()

# print(PbModel(model_path).predict(20))