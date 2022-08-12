#Python函数是对象，是function类的实例
"""
一等对象：
在运行时创建
能赋值给变量或数据结构中的元素
能作为参数传给函数
能作为函数的返回结果

函数也是一等对象
"""
def f1(n):
    '''函数说明文档'''
    return n+10

print(type(f1))
print(f1.__doc__)

f1_var = f1
print(f1_var(5))
l = list(map(f1,[1,2,3,4]))
print(l)

#因为函数是对象，你甚至可以为其设定属性
f1_var.short_description = "receive one number and plus 10 on it"
print(f1_var.short_description)

#函数具有普通类对象不具有的属性
class C:pass
obj = C()
def func():pass
print(set(dir(func)) - set(dir(obj)))#用集合做差集




