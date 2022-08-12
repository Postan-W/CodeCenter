#参考链接https://docs.python.org/zh-cn/3/library/argparse.html
"""
    name or flags - 一个命名或者一个选项字符串的列表，例如 foo 或 -f, --foo。

    action - 当参数在命令行中出现时使用的动作基本类型。

    nargs - 命令行参数应当消耗的数目。

    const - 被一些 action 和 nargs 选择所需求的常数。

    default - 当参数未在命令行中出现并且也不存在于命名空间对象时所产生的值。

    type - 命令行参数应当被转换成的类型。

    choices - 可用的参数的容器。

    required - 此命令行选项是否可省略 （仅选项可用）。

    help - 一个此选项作用的简单描述。

    metavar - 在使用方法消息中使用的参数值示例。

    dest - 被添加到 parse_args() 所返回对象上的属性名。

"""
"""

class argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)

    创建一个新的 ArgumentParser 对象。所有的参数都应当作为关键字参数传入。每个参数在下面都有它更详细的描述，但简而言之，它们是：

        prog - 程序的名称（默认：sys.argv[0]）

        usage - 描述程序用途的字符串（默认值：从添加到解析器的参数生成）

        description - 在参数帮助文档之前显示的文本（默认值：无）

        epilog - 在参数帮助文档之后显示的文本（默认值：无）

        parents - 一个 ArgumentParser 对象的列表，它们的参数也应包含在内

        formatter_class - 用于自定义帮助文档输出格式的类

        prefix_chars - 可选参数的前缀字符集合（默认值：'-'）

        fromfile_prefix_chars - 当需要从文件中读取其他参数时，用于标识文件名的前缀字符集合（默认值：None）

        argument_default - 参数的全局默认值（默认值： None）

        conflict_handler - 解决冲突选项的策略（通常是不必要的）

        add_help - 为解析器添加一个 -h/--help 选项（默认值： True）

        allow_abbrev - 如果缩写是无歧义的，则允许缩写长选项 （默认值：True）

        exit_on_error - 决定当错误发生时是否让 ArgumentParser 附带错误信息退出。 (默认值: True)

    在 3.5 版更改: 添加 allow_abbrev 参数。

    在 3.8 版更改: 在之前的版本中，allow_abbrev 还会禁用短旗标分组，例如 -vv 表示为 -v -v。

    在 3.9 版更改: 添加了 exit_on_error 形参。

"""

