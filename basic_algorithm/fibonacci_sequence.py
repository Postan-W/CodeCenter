"""
递归法
时间复杂度分析：
每次调用函数视为一个时间单位。那么递归树的节点个数就是时间复杂度。n出现一次；n-1出现一次，因为只有n才能产生n-1;
n-2出现两次，因为只有前面的一个n和一个n-1才能产生n-2;n-2
"""
k1 = 0
def fibonacci_recurence(n):
    global k1
    if n == 1 or n == 2:
        k1 += 1
        return 1
    else:
        k1 += 2
        return fibonacci_recurence(n-1) + fibonacci_recurence(n-2)

#迭代方法，复杂度线性的比如下面就是3*n
k2 = 0
def fibonacci_iteration(n):
    global k2
    if n == 1 or n == 2:
       return 1
    else:
        a,b = 1,1
        for _ in range(3,n+1):
           c = a + b
           a = b
           b = c
           k2 += 3

        return c

print("递归法结果：%d  复杂度：%d"%(fibonacci_recurence(30),k1))
print("迭代法结果：%d  复杂度：%d"%(fibonacci_iteration(30),k2))
#可以看出迭代法比递归法高效很多


#使用yield关键字
def fibonacci_generator():
    a,b = 0,1
    while True:
        yield b
        a,b = b,a+b
print("========================================================")
f = fibonacci_generator()
for i in range(10):
    print(next(f))