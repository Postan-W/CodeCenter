from mxnet import autograd, nd
from mxnet import gluon
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
import numpy as np
rng = np.random
# define data format
input_dim = 200000
input_num = 100

# prepare label data
true_w = rng.randn(input_dim)
true_b = 7.6

x_label = nd.random.normal(shape=(input_num, input_dim))
y_label = true_w[0] * x_label[:, 0] + true_w[1] * x_label[:, 1] + true_b

# print (x_label)
# print (y_label)

# load input data
batch_size = 10
dataset = gdata.ArrayDataset(x_label, y_label)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# define net(model)
net = nn.HybridSequential() # make it hybrid to optimize computing
net.add(nn.Dense(1))
net.initialize()

# loss and optimize functions
loss = gloss.L2Loss()
optimize_method = "sgd"
learning_rate = 0.03
trainer = gluon.Trainer(net.collect_params(), optimize_method, {"learning_rate": learning_rate})

# train
num_epoch = 20
for epoch in range(0, num_epoch):
    for x, y in data_iter:
        with autograd.record():
            temp_loss = loss(net(x), y)
        temp_loss.backward()
        trainer.step(batch_size)
    print ("epoch %d, loss %f" % (epoch, loss(net(x_label), y_label).mean().asnumpy()))

# the trained parameters
print (net[0].weight.data(), net[0].bias.data())

# # test the model
# x_test = nd.array([[3, 5], [6, 10], [13, 7]])
# net(x_test)

# export net json and param
net.hybridize()
# Please first call block.hybridize() and then run forward with this block at least once before calling export.
net(x_label)
net.export("simple_net", num_epoch)