#slice(start,stop,step),不包含stop,step默认值为1
l = list(range(10))
print(l[::2])
#对切片赋值，右侧必须是可迭代对象，即便是一个值也要表示成可迭代对象
#step为1时
l[2:8] = [0,0]#长度小于切片范围，则index为4,5,6,7的原有元素被抛弃
print(l)
l = list(range(10))
l[2:4] = [0,0,0,0,0]#长度大于切片范围也不影响赋值，范围之外的原位置元素的index往后延
print(l)
#step大于1时，可迭代对象的长度必须和切片元素个数一致
l = list(range(10))
l[2:8:2] = [0,0,0]
print(l)
del l[2:7]
print(l)

print([1,2,3]*3)
