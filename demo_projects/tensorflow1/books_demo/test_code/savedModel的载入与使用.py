import tensorflow as tf
print(tf.__version__)
model_path = "../test_code/models2_savedmodel"
#在知道signature信息的情况下

def method1():
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],model_path)#被加载的图就是当前的默认图
        #从下面的代码形式中可以猜测meta_graph.signature_def["xxx"]返回的是一个对象
        input_tensor_name = meta_graph.signature_def["graph_information"].inputs["model_input"].name
        output_tensor_name = meta_graph.signature_def["graph_information"].outputs["model_output"].name

        graph = tf.get_default_graph()

        input_tensor = graph.get_tensor_by_name(input_tensor_name)
        output_tensor = graph.get_tensor_by_name(output_tensor_name)

        print(sess.run(output_tensor,feed_dict={input_tensor: 20}))


#不知道变量名称的情况下
def method2():
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                                model_path)  # 被加载的图就是当前的默认图
        print("meta_graph的类型是:",type(meta_graph))

        signature_def = meta_graph.signature_def

        my_signature = meta_graph.signature_def["graph_information"]

        print(type(signature_def))

        signature_def = list(signature_def.items())#与此相反的操作是dict[(k1,v1),(k2,v2),...]

        signature_def = signature_def[0][1]#跟my_signature是等效的，这是因为只有一个键值对(graph_information,value)

        print(signature_def)

        #下面就是取到的输入输出张量的对应key
        signature_def_inputs = list(signature_def.inputs.items())[0][0]
        signature_def_outputs = list(signature_def.outputs.items())[0][0]
        #上面两句可能有些怪，其实就是解析字典，方法有多种，比如list(signature_def.inputs)就等价于list(signature_def.inputs.items())[0][0]
        print(signature_def_inputs,type(signature_def_inputs))
        print(signature_def_outputs, type(signature_def_outputs))


        #接下来与method1就一样了
        graph = tf.get_default_graph()

        input_tensor = graph.get_tensor_by_name(signature_def.inputs[signature_def_inputs].name)
        output_tensor = graph.get_tensor_by_name(signature_def.outputs[signature_def_outputs].name)

        print(sess.run(output_tensor, feed_dict={input_tensor: 20}))

#使用张量的名称还原输入输出
def method3():
    with tf.Session() as sess:
        meta_graph = tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],model_path)
        graph = tf.get_default_graph()

        input_tensor = graph.get_tensor_by_name("model_input:0")
        output_tensor = graph.get_tensor_by_name("add:0")

        print(sess.run(output_tensor, feed_dict={input_tensor: 20}))

method3()
