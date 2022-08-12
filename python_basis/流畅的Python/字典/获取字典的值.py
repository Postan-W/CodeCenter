l = [('two', 2), ('one', 1), ('three', 3)]
dict_comprehension = {word:number for word,number in l}
four = dict_comprehension.get('four',4)#不改变字典
print(four,dict_comprehension)
three = dict_comprehension.setdefault('three')
print(three)
four = dict_comprehension.setdefault('four',4)#改变字典
print(four,dict_comprehension)
#又因为setdefault返回值即是设定的那个默认值，所以可以在返回值的基础上直接修改默认值
dict_comprehension.setdefault('five',[]).append(5)
print(dict_comprehension)
six = dict_comprehension.setdefault('six',{})
six['value'] = 6
print(dict_comprehension)

