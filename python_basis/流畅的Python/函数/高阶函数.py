#接收函数作为参数或者把函数作为返回值的函数是高阶函数
#比如sorted的key参数用于提供一个函数，把该函数的返回值作为对原元素的排序依据(数值就是比大小，字符串就是比先后顺序)
fruits = ['strawberry', 'fig', 'apple', 'cherry', 'raspberry', 'banana']
fruits = sorted(fruits,key=len)#接收len()函数
print(fruits)
"""
任何单参数函数都能作为 key 参数的值。例如，可以把各个单词反
过来拼写的情形作为排序条件。注意，列表里的单词没有变，我们只是把反向拼写当作
排序条件，因此各种浆果（berry）都排在一起
"""
def reverse(word):
    return word[::-1]
fruits = sorted(fruits,key=reverse)
print(fruits)

#lambda函数，也叫匿名函数
fruits = sorted(fruits,key=lambda word:word[::-1])
print(fruits)

#dir也是高阶函数,返回参数的属性、方法列表
print(dir(reverse))

