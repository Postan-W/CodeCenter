"""
Date: 2022/04/26
"""
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from base_config import *
from pyspark.ml.classification import LogisticRegression,DecisionTreeClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer,IndexToString,VectorAssembler
from pyspark.ml import Pipeline,PipelineModel
import pandas as pd
import numpy as np

data = spark.createDataFrame(pd.DataFrame({'feature1':[i for i in range(100)],'feature2':[i for i in range(100)],'feature3':[i for i in range(100)],'feature4':np.random.randint(0,5,size=(100))}))
data.show(10)
def generate_model():
    vector_assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")  # 构建特征列
    lr = LogisticRegression(featuresCol="features", labelCol="feature4", maxIter=100, regParam=0.3,
                            elasticNetParam=0.8)  # 预测结果默认是放到prediction列，所以不用指定结果列
    train, test = data.randomSplit([0.7, 0.3])
    pipeline = Pipeline(stages=[vector_assembler, lr])
    pipeline_model = pipeline.fit(train)
    predictions = pipeline_model.transform(test)
    predictions.select("feature4", "prediction").show(10, truncate=False)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="feature4")
    accuracy = evaluator.evaluate(predictions)
    print("准确率：", accuracy)
    # 获取logistic模型信息
    lr_model = pipeline_model.stages[2]
    print("coefficient:", lr_model.coefficientMatrix)
    print("intercept:", lr_model.interceptVector)
    print("类别数:{}；特征数:{}".format(lr_model.numClasses, lr_model.numFeatures))

generate_model()



