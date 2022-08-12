"""
spark.mllib是旧版本spark基于RDD的机器学习库；spark.ml是新版本基于DataFrame的机器学习库
Transformer:转换器。调用transform方法将操作dataframe得到新的dataframe，在表现上，新的dataframe
是在原有的dataframe列基础上增加了列。比如模型也是一个转换器。
Estimator:评估器。调用fit方法接收DataFrame，生成Transformer(一般是模型)。fit不一定等同于train，比如特征转换.py中使用的StringIndexer
使用fit(dataframe)相当于适配这个dataframe，获取到了dataframe的category列各个标签的频率以及为其设定索引，之后作为转换器的model
所transform的仍然是这个dataframe，作用是将fit来的信息真正作用到dataframe上生成新列。fit和transform如果使用不同的dataframe是不合理的，
比如transform的dataframe中有fit时用到的df不包含的标签值，不但理论上把一个df的信息作用到另一个上面不合理，并且这种情况会直接报错
Parameter:参数。用于设置转换器和评估器的参数
Pipeline：流水线。将转换或评估的过程串联。包含n个pipelineStage，即n个转换器或评估器。调用pipeline的fit方法生成pipeline模型,
则pipeline也可以视为评估器用来生成的PipelineModel即为转换器
"""
#案例。训练logistic分类模型判别句子中是否包含spark
from base_config import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF,Tokenizer

data = spark.createDataFrame([(0,"a b c d e spark",1.0),(1,"d f s",0.0),(2,"spark d ga s",1.0),(3,"g g spark da",1.0)],schema=["id","text","label"])
tokenizer = Tokenizer(inputCol="text",outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10,regParam=0.001)
#模型接收的特征列和标签列的默认名称为features和label
pipeline = Pipeline(stages=[tokenizer,hashingTF,lr ])
model = pipeline.fit(data)#应该注意的是，lr此时也变成了lr的model，即一个转换器
test_data = spark.createDataFrame([(4,"spark is here"),(5,"a b c d e"),(6," hadoop and spark"),(7,"g daf s s")],["id","text"])
prediction = model.transform(test_data).select(["text","probability","prediction"])
prediction.show()