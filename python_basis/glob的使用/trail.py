import glob
"""
1. " * " 匹配0个或多个字符
2. " ? " 匹配单个字符
3. " [] "匹配指定范围内的字符， 例如 ： [0-9]
"""
#返回指定目录(绝对或相对路径都行)下的满足筛选条件的所有文件路径(按照相对路径查找，返回的就是相对路径；绝对路径同此)，不包含子目录下的内容，类型是列表。
files = glob.glob("./[123].txt")
print(files)
print(glob.glob("C:\\Users\\15216\\Desktop\\projects\\CodeCenter\\python_basis\glob的使用\*"))
#glob.iglob与glob的区别是，前者返回的是generator，后者返回的是list
files = glob.iglob("C:\\Users\\15216\\Desktop\\projects\\CodeCenter\\python_basis\glob的使用\*")
print(files)
print(list(files))
