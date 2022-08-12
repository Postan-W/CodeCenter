"""
@Time : 2021/4/25 9:50
@Author : wmingzhu
@Annotation : 
"""
"""
使用xml.dom解析xml
文件对象模型（Document Object Model，简称DOM），是W3C组织推荐的处理可扩展置标语言的标准编程接口。
一个 DOM 的解析器在解析一个 XML 文档时，一次性读取整个文档，把文档中所有元素保存在内存中的一个树结构里，
之后你可以利用DOM 提供的不同的函数来读取或修改文档的内容和结构，也可以把修改过的内容写入xml文件。
"""
#因为是模型文件，有固定的解析格式
import xml.dom.minidom as xmldom
import logging
local_path = "../pmml_files/lightgbm_video_20200505_old.pmml"
local_path2 = "../pmml_files/decision_tree1.pmml"
local_path3 = "../pmml_files/part-00000"
def parse_xml(path):
    xml_tree = xmldom.parse(path)
    nodes = xml_tree.documentElement
    datafield_list = nodes.getElementsByTagName("DataField")
    datafield_info = []
    for datafield in datafield_list:
        try:
            datafield_info.append((datafield.getAttribute("name"), datafield.getAttribute("dataType")))
        except Exception as e:
            datafield_info = "请输入正确的字段"
            logging.info(e)
            return datafield_info
    return datafield_info
#
# datafield_list = parse_xml(local_path3)
# print(datafield_list)
# print(len(datafield_list))