import torch
import numpy as np
#hwc=2,3,3
t = torch.Tensor([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
t = np.array([[[1,2,3],[4,5,6],[7,8,9]],[[10,11,12],[13,14,15],[16,17,18]]])
print(t.transpose(2,0,1))
print(True in [True,False,False])
