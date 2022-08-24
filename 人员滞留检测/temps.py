import numpy as np
arr = np.arange(20)
arr1 = arr.reshape((5,2,2))
print(arr1)
arr2 = arr1.transpose((2,0,1))
print(arr2,arr2.shape)
