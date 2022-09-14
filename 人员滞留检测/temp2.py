import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,4'
print(torch.cuda.device_count())
print("cuda:{}".format(",".join([str(i) for i in list(range(len(os.environ['CUDA_VISIBLE_DEVICES'].split(","))))])))
