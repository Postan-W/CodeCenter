from sklearn import svm
from sklearn import datasets
from sklearn.cluster import SpectralClustering
import pickle
import json
import joblib
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession, Row, SQLContext
conf = SparkConf().setAppName("miniProject").setMaster("local[*]")
# sc=SparkContext.getOrCreate(conf)
spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
sc = spark.sparkContext

def model_demo():
    svm_model = svm.SVC()
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    svm_model.fit(X, y)
    return svm_model


#sklearn模型对象通过spark保存的sc.parallelize([model]).saveAsPickleFile(full_path)，仍然通过spark读取
# model = model_demo()
# full_path = "file:///C:\\Users\\15216\\Desktop\\projects\\CodeCenter\\topics\\sklearn\\models\\svmmodel"

save_by_spark = lambda model,full_path:sc.parallelize([model]).saveAsPickleFile(full_path)
load_by_spark = lambda full_path:sc.pickleFile(full_path, 3).collect()[0]
model = load_by_spark("file:///C:\\Users\\15216\\Desktop\\projects\\CodeCenter\\topics\\sklearn\\models\\pujulei\\model\\metadata")
# print(model.labels_)
# print(model)
spc_model = SpectralClustering(n_clusters=model.n_clusters,assign_labels=model.assign_labels,coef0=model.coef0,
                               degree=model.degree,n_init=model.n_init,n_neighbors=model.n_neighbors,gamma=model.gamma)

data = "[[1,2,3,4],[5,6,7,8],[2,4,6,8],[4,8,12,16],[8,16,23,32]]"
data = json.loads(data)
# print(model.affinity_matrix_.shape)
# print(model.affinity_matrix_)
result = list(spc_model.fit_predict(data))
result = [int(e) for e in result]
print(result)
result = {"data":result}
json.dumps(result)

#方法一，使用dumps和loads,但没有存入磁盘
# s = pickle.dumps(clf)
# clf2 = pickle.loads(s)
# print clf2.predict(X[0:1])



# 第二种方法
# dump和load 函数能一个接着一个地将几个对象转储到同一个文件。随后调用 load() 来以同样的顺序检索这些对象
# output = open('D:\\xxx\\data.pkl', 'wb')
# input = open('D:\\xxx\\data.pkl', 'rb')
# s = pickle.dump(clf, output)
# output.close()
# clf2 = pickle.load(input)
# input.close()
# print clf2.predict(X[0:1])

# 第三种方法
# 使用joblib替换pickle，这对大数据更有效，但只能持久化到磁盘
# joblib.dump(clf, 'D:\\xxx\\data.pkl')#也可以使用文件对象
# clf = joblib.load('D:\\xxx\\data.pkl')
