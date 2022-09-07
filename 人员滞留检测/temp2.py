import numpy as np
source = [12,34,25,34,17,34]
condition = [1,22,33,57,3,5]

def get_unique(source_list):
    result = []
    for element in source_list:
        if not element in result:
            result.append(element)
    return result

def filter_the_result(source:list,condition:list):
    #注意len(source)==len(condition)
    unique_list = get_unique(source)
    source = np.array(source)
    all_equal_list = []
    for num in unique_list:
        equals = source == num
        equal_list = []
        for i,equal in enumerate(equals):
            if equal:
                equal_list.append(i)
        all_equal_list.append(equal_list)

    index_to_reserve = []
    for ele in all_equal_list:
        index_to_reserve.append(ele[np.argmin(np.array(condition)[ele])])
    return sorted(index_to_reserve)
print(filter_the_result(source,condition))


