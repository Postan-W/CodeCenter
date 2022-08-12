#特殊方法即类中前后都带有双下划线的方法，又叫魔术方法magic或者双下方法dunder,特殊方法是解释器自动调用，而不是代码显式调用
import collections
from random import choice
from math import hypot

#使用namedtuple创建一个简单的类，它只有属性而没有方法
Card = collections.namedtuple("Card",['rank','suit'])
class Cards:
    ranks = [str(n) for n in range(2,11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank,suit) for rank in self.ranks for suit in self.suits]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, item):
        return self._cards[item]

def cards_test():
    # 索引
    cards = Cards()
    print(choice(cards))  # choice函数会随机返回序列的一个元素
    # 切片
    print(cards[:3])
    print(cards[3::12])  # 起始:结束:步长

    # 迭代
    print("-----------------------------")
    for card in cards:  # 这里是隐式调用__iter__函数，但是没有定义，所以还是调用了__getitem__
        print(card)
    print("-----------------------------")
    # 反迭代
    print("-----------------------------")
    for card in reversed(cards):
        print(card)
    print("-----------------------------")
    # 上面操作的本质都是把item索引值传给__getitem__函数

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    #如果没有实现 __repr__，当我们在控制台里打印一个对象时，得到的字符串类似为<Vector object at 0x10e100070>
    def __repr__(self):
        return 'Vector(%r, %r)' % (self.x, self.y)

    def __abs__(self):
        return hypot(self.x, self.y)

    def __bool__(self):
        return bool(abs(self))

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Vector(x, y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

def vector_test():
    vector1 = Vector(3,4)
    print(vector1)
    print(abs(vector1))
    print(vector1*3)
    print(vector1+vector1)
    if vector1:#python会调用bool函数，即等价于bool(vector1)。bool(x) 的背后是调用 x.__bool__()
        print("布尔")
        print(bool(vector1))

vector_test()