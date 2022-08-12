#本脚本用于在项目目录下筛选除了.idea和.git以外的大于一定体积(以Mb为单位)的文件
import os

def hidden_layer(current_dir,targets,threshold):
    elements = os.listdir(current_dir)
    for element in elements:
        #当然，这里的current_dir都是不带/结尾的，这里只是规范写法
        full_path = current_dir + element if current_dir.endswith("/") else current_dir + "/" + element
        if not os.path.isdir(full_path):
            size = round(os.stat(full_path).st_size/1024/1024,5)
            if size >= threshold:
                targets.append(("{}Mb".format(size),full_path))
        else:
            hidden_layer(full_path,targets,threshold)


def input_layer(threshold:float,root_path)->list:
    '''
    :param threshold: 筛选文件的大小，单位Mb
    :param root_path: 目标目录
    :return:包含筛选目标的列表,[(文件大小，文件路径),.....]
    '''
    elements = os.listdir(root_path)
    targets = []
    try:
        elements.remove(".idea"), elements.remove(".git")
    except:
        print("不存在.git和.idea")

    for element in elements:
        full_path = root_path + element if root_path.endswith("/") else root_path + "/" + element
        if not os.path.isdir(full_path):
            size = round(os.stat(full_path).st_size/1024/1024,5)
            if size >= threshold:
                targets.append(("{}Mb".format(size),full_path))
        else:
            hidden_layer(full_path,targets,threshold)

    return targets

targets = input_layer(10,"./")
print(targets)
