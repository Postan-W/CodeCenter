import numpy as np
a = np.ones((5,5))
np.save("ones.npy",a)
b = np.load("ones.npy")
print(b)