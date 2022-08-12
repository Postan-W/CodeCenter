import numpy as np

#原始标签和类别数
def one_hot(data,classes):
    length = len(data)
    zeros = np.zeros((length,classes))
    for i in range(length):
        zeros[i][data[i]] = 1
    return zeros
