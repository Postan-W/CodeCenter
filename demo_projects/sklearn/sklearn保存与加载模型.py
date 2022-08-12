from sklearn import svm
from sklearn import datasets
import pickle
import joblib
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

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
