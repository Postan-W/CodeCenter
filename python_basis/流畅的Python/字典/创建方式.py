a = dict(one=1, two=2, three=3)
b = {'one': 1, 'two': 2, 'three': 3}
c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
d = dict([('two', 2), ('one', 1), ('three', 3)])
e = dict({'three': 3, 'one': 1, 'two': 2})

#通过字典推导创建
l = [('two', 2), ('one', 1), ('three', 3)]
dict_comprehension = {word:number for word,number in l}
print(dict_comprehension)



