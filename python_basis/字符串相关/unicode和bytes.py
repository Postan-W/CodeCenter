#Python字符串使用Unicode编码来表示文本。UTF-8是Unicode字符集下的一个具体的变长编码实现

#直接用用16进制表示
print("\u00C6")
#用字符的名称表示
print("This is a cat: \N{Cat}")

"""
bytes是一种比特流，它的存在形式是01010001110这种。
我们无论是在写代码，还是阅读文章的过程中，肯定不会有人直接阅读这种比特流，它必须有一个编码方式，
使得它变成有意义的比特流，而不是一堆晦涩难懂的01组合。因为编码方式的不同，对这个比特流的解读也会不同
"""
s = "中文"
print(type(s))
b = bytes(s, encoding='utf-8')#等同于s.endcoding("utf-8")，也就是说s要变成二进制流了，并且以utf-8的结构将自己对应到二进制形式上
print(b,type(b))
print(s.encode('utf-8'),type(s.encode('utf-8')))
print(b.decode('utf-8'))#以一种编码来表示二进制流，也就是将二进制流以该编码的结构来表示。等同于str(b,encoding='utf-8')
print(str(b,encoding='utf-8'),type(str(b,encoding='utf-8')))
"""
b实际上是一串01的组合，但为了在ide环境中让我们相对直观的观察，它被表现成了b'\xe4\xb8\xad\xe6\x96\x87',
\xe4是十六进制的表示方式(原本是8位二进制数)，它占用1个字节的长度，因此”中文“被编码成utf-8后，我们可以数得出一共用了6个字节，每个汉字占用3个。
字符串类str里有一个encode()方法，它是从字符串向比特流的编码过程。而bytes类型恰好有个decode()方法，
它是从比特流向字符串解码的过程。除此之外，我们查看Python源码会发现bytes和str拥有几乎一模一样的方法列表，
最大的区别就是encode和decode。
"""




