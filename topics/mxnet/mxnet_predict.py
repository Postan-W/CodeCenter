"""
Date: 2022/04/26
"""
import mxnet as mx
import numpy as np
print("Mxnet的版本为:{}".format(mx.__version__))
# define test data
batch_size = 1
num_batch = 5
eval_data = np.array([[3, 5], [6,10], [13, 7]])
eval_label = np.zeros(len(eval_data)) # just need to be the same length, empty is ok
eval_iter = mx.io.NDArrayIter(eval_data, eval_label, batch_size, shuffle=False)

# load model
sym, arg_params, aux_params = mx.model.load_checkpoint("simple_net", 20) # load with net name and epoch num
mod = mx.mod.Module(symbol=sym, context=mx.cpu(), data_names=["data"], label_names=[]) # label can be empty
mod.bind(for_training=False, data_shapes=[("data", (1, 2))]) # data shape, 1 x 2 vector for one test data record
mod.set_params(arg_params, aux_params)

# predict
predict_stress = mod.predict(eval_iter, num_batch)

print (predict_stress) # you can transfer to numpy array