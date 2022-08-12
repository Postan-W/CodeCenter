#在字符串前加r表示原始字符串，里面的特殊字符就被当做普通字符
print("c:\test")#输出c:	est,这里是把\t当做特殊字符处理了
print(r"c:\test")
print(repr("c:\test"))
print(r"c:\tes\"t")#双引号仍然需要转义