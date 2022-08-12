import numpy as np
#PART 1 索引和切片的操作
#PART 1.1 索引的操作。以二维为例
def indexing():
    a = np.random.randint(1, 10, (5, 5))
    print(a)
    print("选取第一个维度的第3个元素")
    print(a[2])
    print("选取第一个维度的第三个元素且第二个维度的第4个元素")
    print(a[2, 3])
    print("嵌套的写法:选取第一个维度的第1,2,3个元素且选择第二个维度的第0,1,2个元素。\n注意第二个维度的各个数值仅作用域第一个维度所选择的对应元素\n,所以"
          "第二个维度的数值个数要与第一个维度的一样。如果第二个维度只有一个数值，\n那么就广播到与第一个维度的数值数量一样")
    print(a[[0, 1, 2], [0, 1, 2]])
    print(a[[0, 1, 2], 2])
    print(a[[0, 1, 2]])
    print("二级索引，本质上就是第一次索引取出的元素再对其索引")
    print(a[0][1])
#PART 1.2 切片的操作。以二维为例
def slicing():
    a = np.random.randint(1,10,(5,5))
    print(a)
    print("起始、结束、步长。其中步长默认为1，所以一般只用一个colon")
    print(a[:])
    print("步长为2")
    print(a[::2])
    print("带上起止")
    print(a[1:4:2])
    print("多个维度的切片用逗号隔开")
    print(a[1:4:2,0:4:2])

#PART 2 关于索引和切片返回值的分析
#PART 2.1 indexing return
def indexing_return():
    a = np.random.randint(1, 10, (5, 5))
    print(a)
    b = a[[0, 1, 2], [0, 1, 2]]
    print("具体到一个数值，索引的返回数值的复制给新对象，和原数组没有任何关系")
    print(b)
    print(id(a),id(b))
    b[0] = 10000
    print(a)
    print(b)
    print("分析子数组")
    c = a[[0,1,2]]
    print(c)
    c[0][0] = 10000
    print(a)
    #从试验看索引得到的对象和原对象完全独立，没有重叠

def slicing_return():
    a = np.random.randint(1, 10, (5, 5))
    print(a)
    b = a[::2]
    b[0][0] = 10000
    print("-------------------------------------------------")
    print(a)
    print(id(a),id(b))
    print(id(a[0]),id(b[0]))
    #从实验看，虽然切片得到的对象和原对象是两个不同的对象，但切片的子元素却是一样的，或者说子元素就是用的同一个内存空间的数据






