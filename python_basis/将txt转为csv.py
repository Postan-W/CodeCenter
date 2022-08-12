
txt_path = "./temp_files/data.txt"
#波长数共2049种，所以奇数行的数值个数一定要为2049个,其中最后一个是换行符，要去掉，即有效的是2048
def check(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if (i+2)%2==0 and len(lines[i].split(" ")) != 2049:
                print("异常")
    return lines


lines = check(txt_path)
wavelength = lines[0].split(" ")[:2048]
time = []
intensity = []
for i in range(1,len(lines)):
    if i%2 == 1:
        time.append(lines[i].strip())
    else:
       intensity.append([int(value) for value in lines[i].split(" ")[:2048]])

import pandas as pd
import numpy as np
dataframe = pd.DataFrame()
intensity2 = np.array(intensity)
#构造time列
dataframe["time"] = time
for i in range(len(wavelength)):
    dataframe[wavelength[i]] = intensity2[:,i]

dataframe.to_csv("target.csv")

