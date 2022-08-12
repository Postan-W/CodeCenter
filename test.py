def f1(f):
    def f_inner(a,b):
        return f(a+b)
    return f_inner

@f1
def f2(number):
    return number*number

print(f2(3,5))#调用被装饰函数实际上是调用被装饰后返回的可调用对象