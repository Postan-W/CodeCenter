#汉字码表https://blog.csdn.net/u010811143/article/details/51560246
cafe = bytes('café美', encoding='utf_8')
print("汉字{}的码点为:{}".format(chr(ord('美')),ord('美')))
"""
cafe的长相为b'caf\xc3\xa9\xe7\xbe\x8e'
bytes 或 bytearray 对象的各个元素是介于 0~255（含）之间的整数(用x即16进制表示)，因为前面三个caf在utf-8中和ASCII一样都只占一个字节，
所以是属于可打印的字节；而é和美分别占2个和三个字节，即各代表bytes对象的2和3个元素，即在utf-8编码下占的，2和3个字节，又因为无论是
字符的首字节还是非首字节都是1开头的，使得其一定大于127，于是不在可打印的范围内(除ASCII码0～31及127(共33个)是控制字符或通信专
用字符，剩余的都属于可打印字符)，于是显示的是字节的原值。
上面汉字'美'的utf-8编码十六进制形式是\xe7\xbe\x8e，即11100111 10111110 10001110，
因为原始的码点是7f8e(即上面ord('美打印的32654'))，即0111 1111 1000 1110 和上面完全一致
"""
print(len(cafe))
print(cafe[-1])#即x8e

cafe_arr = bytearray(cafe)
print(cafe_arr)
#bytes 是不可变的 bytearray 是可变的，就类似元组和列表的关系
# cafe[0] = 120
cafe_arr[0] = 240
print(cafe_arr)

#bytes或bytearray本质上是数字数组，从创建上也可以看出来
b = bytes([123,124,125,126,255])
print(b,len(b))