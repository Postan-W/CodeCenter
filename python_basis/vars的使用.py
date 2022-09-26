
g1 = 10
g2 = 20

class Person:
    def __init__(self,name,age):
        self.name = name
        self.age = age

person = Person(name="wmingzhu",age=27)
def f(a,b,c):
    person2 = Person(name="wmingzhu", age=27)
    print(vars())#不带参数时返回当前范围的所有局部变量和其值的dict。效果同locals()
    print(locals())
    #注意vars()和locals()得到的只是局部命名空间中对象的拷贝
    print(globals())#输出全局namespace，不是拷贝

    print(vars(person))#返回对象属性和其值的dict

f(1,2,3)