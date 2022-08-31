l = [[1,2,3]]
l2 = [[4,5,6]]
import torch
t1 = torch.tensor(l)
t2 = torch.tensor(l2)
print(torch.cat((t1,t2),0))