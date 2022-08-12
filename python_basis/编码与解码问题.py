"""
@Time : 2021/5/24 11:18
@Author : wmingzhu
@Annotation : 
"""
#python3的print函数默认是按utf-8打印内容的

#下面是GBK编码的内容，含中文
a = b'{"data":[{"\xbd\xda\xb5\xe3":"mn1","\xb8\xfc\xd0\xc2\xca\xb1\xbc\xe4":"2021/2/26  11:10:02","\xc1\xac\xbd\xd3\xca\xfd":"100"}, {"\xbd\xda\xb5\xe3":"mn2","\xb8\xfc\xd0\xc2\xca\xb1\xbc\xe4":"2021/2/26  11:10:02","\xc1\xac\xbd\xd3\xca\xfd":"80"}, {"\xbd\xda\xb5\xe3":"mn3","\xb8\xfc\xd0\xc2\xca\xb1\xbc\xe4":"2021/2/26  11:10:02","\xc1\xac\xbd\xd3\xca\xfd":"80"}]}'


b = a.decode('gbk')#这句话的意思是用gbk的codec即编解码器来解码这段，解码的结果是Unicode编码的字符
print(b)

#c的中文就是Unicode编码的，用print可以直接打印出中文
c = u'{"data": [{"\u8282\u70b9": "mn1", "\u66f4\u65b0\u65f6\u95f4": "2021/2/26\u00a0 11:10:02", "\u8fde\u63a5\u6570": "100"}, {"\u8282\u70b9": "mn2", "\u66f4\u65b0\u65f6\u95f4": "2021/2/26\u00a0 11:10:02", "\u8fde\u63a5\u6570": "80"}, {"\u8282\u70b9": "mn3", "\u66f4\u65b0\u65f6\u95f4": "2021/2/26\u00a0 11:10:02", "\u8fde\u63a5\u6570": "80"}]}'
print(c)

#使用gbk的编解码器来编码
d = "hello".encode('gbk')
print(d)

