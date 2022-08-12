from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from base_config import *
from pyspark.ml.classification import LogisticRegression,DecisionTreeClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer,IndexToString,VectorAssembler
from pyspark.ml import Pipeline,PipelineModel

origin_rdd = sc.textFile("file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\sparkML\\iris.txt")
all_lines = origin_rdd.collect()
header = all_lines[0].split(",")#文件的每一行都被保存在Row对象的value属性里面。这里获取表头
print(header)
data = sc.parallelize(all_lines[1:]).map(lambda r:r.split(",")).map(lambda p:Row(sepal_length=float(p[0]),sepal_width=float(p[1]),
                                                                                                  petal_length=float(p[2]),petal_width=float(p[3]),target=p[4]))

print(data.collect())
data = spark.createDataFrame(data)

def generate_model():
    vector_assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")  # 构建特征列
    string_indexer = StringIndexer(inputCol="target", outputCol="targetIndexed").fit(data)  # 构建索引化的标签列
    print(string_indexer.labels)
    lr = LogisticRegression(featuresCol="features", labelCol="targetIndexed", maxIter=100, regParam=0.3,
                            elasticNetParam=0.8)  # 预测结果默认是放到prediction列，所以不用指定结果列
    # 将预测结果转为字符串标签。string_indexed.labels列表中标签的位置顺序即对应索引，由此还原。已经我验证
    predictions_to_string = IndexToString(inputCol="prediction", outputCol="predictions_string",
                                          labels=string_indexer.labels)
    train, test = data.randomSplit([0.7, 0.3])
    pipeline = Pipeline(stages=[vector_assembler, string_indexer, lr, predictions_to_string])
    pipeline_model = pipeline.fit(train)
    predictions = pipeline_model.transform(test)
    predictions.select("targetIndexed", "prediction").show(10, truncate=False)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="targetIndexed")
    accuracy = evaluator.evaluate(predictions)
    print("准确率：", accuracy)
    # 获取logistic模型信息
    lr_model = pipeline_model.stages[2]
    print("coefficient:", lr_model.coefficientMatrix)
    print("intercept:", lr_model.interceptVector)
    print("类别数:{}；特征数:{}".format(lr_model.numClasses, lr_model.numFeatures))
    print("保存模型")
    pipeline_model.save("file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\sparkML\\logisticClassification")

def load_model():
    model = PipelineModel.load("file:///C:\\Users\\15216\\Desktop\\项目\\PracticeHub\\spark\\sparkML\\logisticClassification")
    model.transform(data).select("prediction","targetIndexed","target").show(10,truncate=False)

load_model()




