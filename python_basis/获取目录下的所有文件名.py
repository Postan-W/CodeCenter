#Author:Mingzhu W
#获取目录下的所有文件名并用目录名作为分割
import os
directory = ""
def get_all_files(directory:str,files:list)->list:
    first_layer = os.listdir(directory)
    for ele in first_layer:
        entire_path = os.path.join(directory, ele)
        if not os.path.isdir(entire_path):
            files.append(ele)
        else:
            files.append("--------------------------"+ele+"-------------------------------")
            files = get_all_files(entire_path,files)
    return files



