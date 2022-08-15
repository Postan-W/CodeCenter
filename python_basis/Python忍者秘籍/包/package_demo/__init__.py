print("每当导入某个包时就会执行包的__init__.py模块;")
import os
import sys

"""
该模块在有关于该包的一切导入时都会执行；
__all__是个列表，它指定了from package import *即模糊导入时导入哪些模块，可以是__init__.py中import的模块，
可以是package下面的模块(或包),如果没有定义该变量，那么模糊导入时只会导入__init__.py中import的模块。
注意__all__是一个模块(或包)列表，它对from package.module import *不起作用
"""
# __all__ = ["in_the_package","in_the_package3","os","sys"]
