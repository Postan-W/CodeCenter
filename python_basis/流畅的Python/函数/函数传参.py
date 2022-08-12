#详细参考:https://blog.csdn.net/Hardworking666/article/details/111709764
"""
必备参数、默认参数、不定长参数是针对形参来讲的。
位置参数和关键字参数都是针对实参来讲的，区别在于传实参的时候有没有指定参数名。
*args：接收多传入的位置参数，以tuple的形式保存;**kwargs：接收多传入的关键字参数，以dict的形式保存。
传参的时候也可以使用*或者**这样的做法，但这属于是对序列或者字典的拆解，跟函数定义中*或**不定长参数的含义不同。

"""
#必备参数
def f_required(a,b,c):
    print(a,b,c)

# f_required(1,2,a=3)错误,a被复制两次
# f_required(b=3,1,2)Positional argument after keyword argument
#默认参数
#必要参数不能位于默认值参数之后
def f_default(a,b,c=5):pass


def f2(*args,**kwargs):#*args必须在**kwargs之前
    print(type(args),type(kwargs))
    print(len(args),len(kwargs))

f2(1,2,3,4,name="mingzhu",height=185)
f2([1,2,3,4],name="mingzhu",height=185)
f2(*[1,2,3,4],{"name":"mingzhu","height":185})
f2(*[1,2,3,4],**{"name":"mingzhu","height":185})
#总之，位置参数就作为args的一个参数，一个指定名字的参数就作为kwargs的key和value

# f2(*[1,2,3,4],name="明珠",5)#Error,Positional argument after keyword argument
def f3(*args,date="xxx",location,**kwargs):
    print(type(args),type(kwargs))
    print(len(args),len(kwargs))

f3(date="xxx",location="beijing",*[1,2,3,4],**{"name":"mingzhu","height":185})
f3(*[1,2,3,4],"xxx",location="beijing",**{"name":"mingzhu","height":185})

#上面说过必备参数要在默认值参数之前，这里的location不叫必备参数，而叫做仅限关键字参数，在*之后，即必须指名传参，不然报错
def f4(test,date="xxx",*args,country="china",location,job,**kwargs):
    print("====================================")
    print(test)
    print(date)
    print(args)
    print(location)
    print(job)
    print(kwargs)

f4("test","2022",1,2,3,location="beijing",job="farmer",**{"name":"mingzhu","height":185})
f4("test",location="beijing",job="farmer",**{"name":"mingzhu","height":185})
"""
综上：
f4里面包含了python3所有形参类型f4(必备参数,默认值参数,不定位置参数,仅限关键字参数,不定关键字参数);
*args之前的参数传参和不包含不定参数的函数传参一样，即对形参来说，必备参数要在默认值参数之前，这样的话传实参时，
如果实参个数只等于默认值那么就依次把实参赋给必备参数，默认值参数保持默认值，否则的话就一次覆盖默认值，对实参来说，
关键字参数要放在位置参数之后,因为位置参数是按位置依次赋的，如果插进去一个关键字参数，那么按位置赋值就存在商议性，因为很可能重复赋值，
关键字参数之间的顺序不用保持形参时的顺序。
从*开始所有的位置参数都被放入到了args里面，其后的参数必须指定关键字来传参(除非有默认值，可以不传该参数)，目的是和
放在参数列表最后的**kwargs作区别
"""