"""
在Python中没有块的概念，不像C，花括号围起来的区域为块域，比如
if(m > 10){
    int b = 100;
}
上面定义的b在块外面是访问不到的。
而在Python中区分局部和全局就是看在函数内部还是外部.
"""
if 4 < 10:
    a = 100;

print("可以访问到a:{}".format(a))
