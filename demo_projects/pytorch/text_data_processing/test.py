import torch
a = torch.ones(2,2)
print(a)
b = a.view((-1,2,1))#参数中的-1就代表这个位置由其他位置的数字来推断，所以结果等同于a.view((2,2,1))
print(b)
c = a.view((2,2,-1))
print(c)