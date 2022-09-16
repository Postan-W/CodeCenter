from pathlib import Path,PurePath,PurePosixPath
import os

def demo1():
    print(__file__)  # 当前文件，C:/Users/15216/Desktop/projects/CodeCenter/python_basis/pathlib模块的使用.py
    file = Path(__file__).resolve()
    print(file.name)#文件名称
    print(file.stem)#名字不带后缀
    print(file.suffix)#后缀，包含.,如.py
    parents = file.parents  # 获取所有层次的父目录
    print(list(parents))
    parent = file.parent  # 上一级父目录
    print(parent)
    print(parent / 'child')  # 直接用斜杠拼接路径
demo1()
def demo2():
    print(Path.cwd())
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[0]
    #os.path.relpath的作用是获取参数1路径相对于参数2路径的相对路径
    print(os.path.relpath(ROOT, Path.cwd()),type(os.path.relpath(ROOT, Path.cwd())))
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
    print(ROOT)#当path为文件夹时，通过yield产生path文件夹下的所有文件、文件夹路径的迭代器
    print(ROOT.is_dir())
    print(list(ROOT.iterdir()))
    print(list(ROOT.parents))#使用.这样的方式找不到父目录

def demo3():
    #purepath只支持对路径字符串本身的操作，实际上不去访问文件操作系统
    path = PurePath(os.getcwd())
    print(PurePath())#无参数时为.即当前路径
    print(path)
    print(path.drive)#获取盘符
    path2 = PurePath("a","b","c")
    print(path2,type(path2))#自动拼接多个字符串为路径。'pathlib.PureWindowsPath'。
    #所在系统是什么就会生成什么风格的path，也可以显示指定
    path3 = PurePosixPath('helloyou','some/path','info')
    print(path3,type(path3))

