## 1.utf-8基本介绍
Unicode包含两层含义，一是它是一个字符集，二是它是一种编码方案，Unicode是国际组织制定的可以容纳世界上所有文字和符号的字符编码方案。

**UCS-2和UCS-4**

Unicode是为整合全世界的所有语言文字而诞生的。任何文字在Unicode中都对应一个值， 
这个值称为代码点（code point）。代码点的值通常写成 U+ABCD 的格式。 而文字和代码点之间的对应关系就是UCS-2（Universal Character Set coded in 2 octets）。 顾名思义，UCS-2是用两个字节来表示代码点，其取值范围为 U+0000～U+FFFF。

为了能表示更多的文字，人们又提出了UCS-4，即代码点的范围扩展至4个字节。 它的范围为 U+00000000～U+7FFFFFFF，其中 U+00000000～U+0000FFFF和UCS-2是一样的。

将码点转换为其他表示形式。UTF（Unicode Transformation Format），其中应用较多的就是UTF-16和UTF-8了。**UTF(Unicode Transformation Format)**
UTF-8是UNICODE的一种变长度的编码表达方式（一般UNICODE为双字节[指UCS2]），UTF-8就是以8位为单元对UCS进行编码，而UTF-8不使用大尾序和小尾序的形式，每个使用UTF-8储存的字符，除了第一个字节外，其余字节的头两个位元都是以"10"开始，使文字处理器能够较快地找出每个字符的开始位置。
为了与以前的[ASCII码](https://so.csdn.net/so/search?q=ASCII码&spm=1001.2101.3001.7020)相容（ASCII为一个字节）以及节省存储空间，UTF-8 选择了使用可变长度字节来储存 Unicode,具体转换关系如下表：

UCS-4                   UTF-8编码
U-00000000 – U-0000007F	0xxxxxxx
U-00000080 – U-000007FF	110xxxxx 10xxxxxx
U-00000800 – U-0000FFFF	1110xxxx 10xxxxxx 10xxxxxx
U-00010000 – U-001FFFFF	11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
U-00200000 – U-03FFFFFF	111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
U-04000000 – U-7FFFFFFF	1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx

为什么第一个字节以110、1110这样的形式开头，而非第一个字节用10开头？

前者是为了使计算机知道要读取的字符占用了几个字节，1的个数对应字节数,并且皆以0结尾是为了跟第一个x位作区分，不然当其为1时就混淆了；

后者是为了容错，例如某一个字符的编码为11100010 11100011 110001111 的首字节错了一位，变成了11000010 11100011 110001111,那么计算机就会只读取两个字节然后接着读取下一个字符，这样就会导致该字符编码是错的及其后面的所有字符的编码都向前移了一个字节，所以后面的所有字符要么编码无法识别，要么不是原本的那个字符，而如果非首字节用10开头，因为10不可能是首字节，所以该可以被跳过，一个字符出错不影响后续的字符。

并且以10开头的话，在字节流的任意一个地方都可以跳过以10开头的然后找下一个字符的开头。
首字节以1xxx开头，非首字节以10开头的另一个好处是直接跳过了可打印字符的范围(0-127),于是某个字符的某个字节不会被当做ASCII
可打印字符而被打印出来

## 2.Unicode转为utf-8

UCS-4(UNICODE)码位	UTF-8字节流
U-00000000 – U-0000007F	0xxxxxxx
U-00000080 – U-000007FF	110xxxxx 10xxxxxx
U-00000800 – U-0000FFFF	1110xxxx 10xxxxxx 10xxxxxx
U-00010000 – U-001FFFFF	11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
U-00200000 – U-03FFFFFF	111110xx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx
U-04000000 – U-7FFFFFFF	1111110x 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx 10xxxxxx

对于0x00-0x7F之间的字符，UTF-8编码与ASCII编码完全相同。UTF-8编码的最大长度是4个字节。从表3-2可以看出，4字节模板有21个x，即可以容纳21位二进制数字。Unicode的最大码位0x10FFFF也只有21位。

**例1：**

“汉”字的Unicode编码是0x6C49。0x6C49在0x0800-0xFFFF之间，使用用3字节模板：1110xxxx 10xxxxxx 10xxxxxx。将0x6C49写成二进制是：0110 1100 0100 1001， 用这个比特流依次代替模板中的x，得到：11100110 10110001 10001001，即E6 B1 89

**例2：**

Unicode编码0x20C30在0x010000-0x10FFFF之间，使用4字节模板：11110xxx 10xxxxxx 10xxxxxx 10xxxxxx。将0x20C30写成21位二进制数字（不足21位就在前面补0）：0 0010 0000 1100 0011 0000，用这个比特流依次代替模板中的x，得到：11110000 10100000 10110000 10110000，即F0 A0 B0 B0



