"""
Python 3 的 str 类型基本相当于 Python 2 的 unicode 类型。
把码位code point转换成字节序列的过程是编码；把字节序列转换成码点的过程是解码
python3.x字符串中的所有字符都是Unicode编码(python2.x 中以Unicode表示的字符串用u‘xxx’，Python3中不用)，即最原始的码点，
因为码点大小不一样则字符所占的字节数是不尽相同的，这会给字符索引带来困难，所以在表示字符时一定都是相同长度的字节数，
具体使用几个字节，是根据字符串中的字符所占用的最大字节数来决定，具体参考Python字符串的底层实现文档

"""
s = 'cafe牛奶'
b = s.encode('utf-8')
print(type(b),len(b))
print(b)
s = b.decode('utf-8')
print(type(s),s)
print(ord('a'))

#会根据字符串中包含的字符情况自动设定编码方式(即是使用几个字节来表示码点)
import sys
print(sys.getsizeof("哈哈") - sys.getsizeof("哈"))
print(sys.getsizeof("aa") - sys.getsizeof("a"))
print(sys.getsizeof("a哈") - sys.getsizeof("哈"))
#可以看出，字符串中一个字符要占用几个字节，是由占用最大字节数的字符决定的
print(sys.getsizeof(None))