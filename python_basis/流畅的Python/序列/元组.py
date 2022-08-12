"""
元组被看作是不可变的列表，因为它作为序列或可迭代对象，用起来感觉和列表一样，只是元素不可变。
"""
#元组拆包的特性是列表所不具备的
def disassemble_tuple():
    #示例1
    a,b,_ = (1,2,3)
    print(a,b,_)
    #示例2
    content = ("Tom","New York",23)
    print("%s住在%s,今年%d岁"%content)
    print(divmod(*(20,8)))
    *some,a,b = range(5)
    print(some,a,b)
    a,*some,b = range(5)
    print(some, a, b)

    city = ("beijing", "cn", 16410, ("pku","tsinghua"))
    #直接拆包
    a,a,a,school = city
    print(school,a)
    #嵌套1拆包
    _,_,_,(school1,school2) = city
    print(school1,school2)

    for e in reversed((1,2,3,4,5)):
        print(e)


disassemble_tuple()
