import os
import shutil
import paddle as paddle
import paddle.fluid as fluid
import time
import numpy as np
paddle.enable_static()

def constant_sum():
    x1 = fluid.layers.fill_constant(shape=[2,2],value=1,dtype='int64')
    x2 = fluid.layers.fill_constant(shape=[2,2],value=1,dtype='int64')
    y1 = fluid.layers.sum(x=[x1,x2])
    place = fluid.CPUPlace()
    exe = fluid.executor.Executor(place)
    exe.run(fluid.default_startup_program())
    result = exe.run(program=fluid.default_main_program(),fetch_list=[y1])
    print(result)

def variable_test():
    a = fluid.layers.create_tensor(dtype='int64',name='a')
    b = fluid.layers.create_tensor(dtype='int64',name='b')
    y = fluid.layers.sum(x=[a,b])
    place = fluid.CPUPlace()
    exe = fluid.executor.Executor(place)
    exe.run(fluid.default_startup_program())
    a1 = np.array([3,2]).astype('int64')
    b1 = np.array([1,1]).astype('int64')
    out_a,out_b,result = exe.run(program=fluid.default_main_program(),feed={a.name:a1,b.name:b1},fetch_list=[a,b,y])
    print(out_a,"+",out_b,"=",result)

def linear_regression():
    x = fluid.data(name='x',shape=[None,1],dtype='float32')
    hidden = fluid.layers.fc(input=x,size=50,act='relu')
    hidden = fluid.layers.fc(input=hidden,size=50,act='relu')
    output = fluid.layers.fc(input=hidden,size=1,act=None)
    infer_program = fluid.default_main_program().clone(for_test=True)
    y = fluid.data(name='y',dtype='float32',shape=[None,1])
    cost = fluid.layers.square_error_cost(input=output,label=y)
    avg_cost = fluid.layers.mean(cost)
    optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01)
    opts = optimizer.minimize(avg_cost)
    place = fluid.CPUPlace()
    exe = fluid.executor.Executor(place)
    exe.run(fluid.default_startup_program())
    #模拟y = 2x + 1的训练数据
    x_data = np.array([[1],[2],[3],[4],[5]]).astype('float32')
    y_data = np.array([[3],[5],[7],[9],[11]]).astype('float32')
    for epoch in range(20):
        train_cost = exe.run(program=fluid.default_main_program(),feed={x.name:x_data,y.name:y_data},fetch_list=[avg_cost])
        print("epoch:%d,Cost:%0.5f"%(epoch,train_cost[0]))

    test_data = np.array([[6]]).astype("float32")
    result = exe.run(program=infer_program,feed={x.name:test_data},fetch_list=[output])
    print("当输入为6时，预测输出为{}".format(result))

linear_regression()