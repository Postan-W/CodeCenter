"""
Date: 2022/03/11
"""
import numpy as np
x = np.random.randint(1,10,(5,5))
print(x)
print("------------------------------")
print(x[::2,::2])
y = x[[3,4]]
print("-------------------------------------------------")
print(y)

