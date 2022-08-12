import random
#可以使用调用运算符()的对象叫可调用对象，使用内置函数callable()可以判断对象内否被调用
print([callable(obj) for obj in [abs,str,100]])
#实现实例方法__call__则可使实例为可调用的
class Mingzhu:
    def __init__(self,*args):
        self.info = args

    def __call__(self, *args, **kwargs):
        index = len(args) - 1
        return self.info[index]

mingzhu = Mingzhu("Gushi","Henan","China")
print(mingzhu.info)
print(mingzhu(1,2,3))
print(callable(mingzhu))

print(dir(mingzhu))

