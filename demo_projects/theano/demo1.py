"""
Date: 2022/04/25
"""
# Demo:theano模型训练、保存及加载
import theano
import numpy as np
import theano.tensor as T
import pickle
import time

print("Theano的版本为：{}".format(theano.__version__))


def compute_accuracy(y_target, y_predict):
    correct_prediction = np.equal(y_predict, y_target)
    accuracy = np.sum(correct_prediction) / len(correct_prediction)
    return accuracy


# 数据由numpy随机生成，训练一个n元线性判别模型
def train():
    rng = np.random
    N = 400  # batch
    feats = 78400  # inputsize
    D = (rng.randn(N, feats), rng.randint(size=N, low=0,high=2))# data和label
    x = T.dmatrix("x")
    y = T.dvector("y")
    w = theano.shared(rng.randn(feats), name="w")
    b = theano.shared(0., name="b")
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
    prediction = p_1 > 0.5
    xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
    cost = xent.mean() + 0.01 * (w ** 2).sum()  # 这里是BCE加上了个L2
    gw, gb = T.grad(cost, [w, b])
    learning_rate = 0.1
    train = theano.function(
        inputs=[x, y],
        updates=((w, w - learning_rate * gw), (b, b - learning_rate * gb)))

    predict = theano.function(inputs=[x], outputs=prediction)
    for i in range(5):
        print("第{}轮".format(i + 1))
        train(D[0], D[1])

    with open('theanomodel{}.pickle'.format(int(time.time())), 'wb') as file:
        model = [w.get_value(), b.get_value()]
        pickle.dump(model, file)
        print(model[0][:10])
        print("accuracy:", compute_accuracy(D[1], predict(D[0])))


def predict():
    rng = np.random
    N = 400  # batch
    feats = 784  # inputsize
    D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))  # data和label
    x = T.dmatrix("x")
    y = T.dvector("y")
    w = theano.shared(rng.randn(feats), name="w")
    b = theano.shared(0., name="b")
    p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
    prediction = p_1 > 0.5
    xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
    cost = xent.mean() + 0.01 * (w ** 2).sum()  # 这里是BCE加上了个L2
    gw, gb = T.grad(cost, [w, b])
    learning_rate = 0.1
    train = theano.function(
        inputs=[x, y],
        updates=((w, w - learning_rate * gw), (b, b - learning_rate * gb)))

    predict = theano.function(inputs=[x], outputs=prediction)
    with open('theanomodel.pickle', 'rb') as file:
        model = pickle.load(file)
        w.set_value(model[0])
        b.set_value(model[1])
        print(w.get_value()[:10])
        print("accuracy:", compute_accuracy(D[1], predict(D[0])))


train()













