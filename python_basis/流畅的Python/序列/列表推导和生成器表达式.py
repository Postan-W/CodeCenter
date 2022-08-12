"""
列表推导list comprehension是构建列表的快捷方式，而生成器表达式generator expression则可以用来创建其他任何类
型的序列。
set tuple dict 等其他序列类型也可以所使用列表推导的形式产生
"""
def list_comprehension():
    #得到字符的Unicode码位
    symbols = '$¢£¥€¤'
    codes = [(char,ord(char)) for char in symbols if ord(char) > 1000]
    print(codes)
    #列表推导中for后面的局部变量和in后面的上下文变量即便同名也不会干扰
    x = "ABC"
    abc_codes = [ord(x) for x in x]
    print(x)
    #实现笛卡尔积，即双重for循环的列表推导式
    str1 = "abc"
    str2 = "def"
    cartesian_product = [(char1,char2) for char1 in str1 for char2 in str2]
    print(cartesian_product)

#生成器表达式使用圆括号，生成器表达式的好处是一次产生一个元素，从而节省内存
