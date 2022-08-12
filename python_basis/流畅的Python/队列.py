"""
使用Python其他的数据结构实现队列或队列从效果上也是可以达到的，比如使用列表的.append()表示进队列,.pop(0)表示出队列,
但是这样效率低，比如.pop(0)出队列意味着后面所有的元素都要移位
"""
#双端队列不是队列，它是在两端都可以添加删除元素的数据结构，从数据进出特点上看它更像是栈
from collections import deque
queue = deque(range(10),maxlen=10)
queue.rotate(3)#把最后n个移动到前面
print(queue)
queue.rotate(-3)#把最前面n个移动到后面
print(queue)
print(len(queue) == queue.maxlen)
queue.appendleft(100)
queue.appendleft(200)
for i in range(10):
    print(queue.pop())#pop是最右端数据出栈，popleft是最左端数据出栈


