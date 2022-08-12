import argparse
#使用 argparse 的第一步是创建一个 ArgumentParser 对象：
parser = argparse.ArgumentParser(prog="ArgumentParser 对象使用 sys.argv[0] 来确定如何在帮助消息中显示程序名称，默认就是py文件名称，即sys.argv[0]",description="计算参数值的和")
#下面参数的名字'integers'等都可以随意取
#输入的参数都默认视为字符串，而参数type可以指定类型
#位置参数
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')
#运行的时候-h或者--help参数可以展示参数的信息，这也是上面函数里help=的作用
parser.add_argument('strings')
parser.add_argument('--image_path',required=True,type=str,help='Picture file path')
#--表示可选参数，运行的时候可以不传。短杠表示缩略写法
parser.add_argument('-s','--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')
parser.add_argument('--min',action='store_const',const=min,dest='minimum')
#在脚本中，通常 parse_args() 会被不带参数调用，而 ArgumentParser 将自动从 sys.argv 中确定命令行参数
import sys
print(sys.argv)
args = parser.parse_args()
print(args)
print(args.integers)
print(args.accumulate(args.integers))