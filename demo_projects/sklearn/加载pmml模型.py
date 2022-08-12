from pypmml import Model
from utility.parse_pmml import parse_xml

path1 = "../../pmml_files/lightgbm_video_20200505_old.pmml"
path2 = "../../pmml_files/part-00000"

def test1(path):
    model = Model.fromFile(path)  # 或者Model.load(xxx)
    # 特征之间使用\t分割，最后一个特征有换行符\n
    with open("./lightgbmtestdata.txt", "r") as f:
        all_lines = f.readlines()
        samples = []
        for line in all_lines:
            features = line.split("\t")
            features[-1] = features[-1].strip('\n')
            samples.append(features)

    test_data = dict(list(zip(samples[0], samples[2])))
    print("输入的数据类型为:", type(test_data))
    print(len(test_data), test_data)
    result = model.predict(test_data)
    result = dict(result)
    print(result, type(result))


def test2(path):
    model = Model.fromFile(path)  # 或者Model.load(xxx)
    feature_list = parse_xml(path2)
    print(feature_list)
    data = {"c_29345": 1, "c_34285": 2.3, "c_IS_OUTNET": 3.3}
    print(model.predict(data))

test1(path1)



