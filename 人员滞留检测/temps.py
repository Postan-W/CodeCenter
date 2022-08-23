import numpy as np
arr = np.arange(30)
print(arr)
arr1 = arr.reshape((5,2,3))
print(arr1)
arr2 = arr1.transpose((2,0,1))
print(arr2,arr2.shape)
