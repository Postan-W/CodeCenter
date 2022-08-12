#安装gin：pip install gin-config。创建.gin文件来保存配置信息
import gin
"""
在方法上申明该装饰器那么方法的所有的参数都可配置化，返回的对象相当于是被修饰的函数对应的参数被设定默认值的版本，
所谓对应的参数，就是.gin文件中设定了值的参数。
配置文件的写法是‘方法名.参数名=值’,其中值可以为任意python支持的类型
"""
@gin.configurable
def gin_function_test(p1,p2,p3):
    print("第一个参数是:{}".format(p1))
    print("第二个参数是:{}".format(p2))
    print("第三个参数是:{}".format(p3))

"""装饰类实际上是装饰类的构造器，和对方法的规则一样。配置文件的写法是‘类名.构造器参数名=值’"""
@gin.configurable
class GinClassTest:
    def __init__(self,p1,p2,p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def print_value(self):
        print("第一个属性值是:{}".format(self.p1))
        print("第二个属性值是:{}".format(self.p2))
        print("第三个属性值是:{}".format(self.p3))

gin.parse_config_file('config.gin')
#因为p2参数已经被配置了相当于有默认值10，这里不给p2赋值也行，如果赋值则覆盖了配置文件中的默认值
gin_function_test(p1=5,p3=15)
gin_function_test(p1=5,p2=20,p3=15)
gct = GinClassTest()
gct.print_value()
