#这里使用的是TF-IDF方法
from base_config import *
from pyspark.ml.feature import HashingTF,IDF,Tokenizer
from pyspark.ml.feature import OneHotEncoder,VectorIndexer,StringIndexer,IndexToString,VectorAssembler
from pyspark.ml.linalg import Vector,Vectors
def tf_idf_vectorize():
    # 一个句子代表TF-IDF中所指的文档
    # A tokenizer that converts the input string to lowercase and then splits it by white spaces.
    sentence_data = spark.createDataFrame(
        [(0, "i heard about spark and i love spark"), (0, "i wish java could use case classes"),
         (1, "logistic regression model is neat")]).toDF("label", "sentence")
    tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
    words_data = tokenizer.transform(sentence_data)
    # words_data.show()
    # 生成特征向量
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2000)
    featurized_data = hashingTF.transform(words_data)
    featurized_data.select("words", "rawFeatures").show(truncate=False)
    idf = IDF(inputCol="rawFeatures", outputCol="features")  # IDF是一个评估器
    idf_model = idf.fit(featurized_data)
    result_vector = idf_model.transform(featurized_data)
    result_vector.select("words", "features").show(truncate=False)
# tf_idf_vectorize()
"""
StringIndexer:可以DataFrame的某一非数值列转为数值列，一般将其应用于对非数值型标签列的数值化中，原理是将出现频率最高的
取值置为0，第二高的置为1，以此类推，所以也叫索引化。如果输入本身就是数值型，其做法是现将数值转为字符，然后再进行索引的操作。
"""
def string_and_indexer():
    # 将标签索引化
    dataframe = spark.createDataFrame([(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"), (6, "d"), (7, "e")],
                                      ["id", "category"])
    indexer = StringIndexer(inputCol="category", outputCol="categoryIndexed")
    indexer_model = indexer.fit(dataframe)
    indexed = indexer_model.transform(dataframe)
    indexed.show()
    # 使用indexToString将用stringToIndex索引化的列转回
    index_to_string = IndexToString(inputCol="categoryIndexed", outputCol="originCategory")#这是一个转换器
    to_string_result = index_to_string.transform(indexed)#说明categoryIndexed列也包含了原始列的信息
    to_string_result.select("originCategory", "category", "categoryIndexed").show()

# string_and_indexer()

def indexer_and_assembler():
    # vectorindexer的作用是将值为向量的列，比如features列的每个值都是包含n个特征的向量,每个特征的取值变化如果小于maxCategories指定的
    # 值，那么就认为其是离散的，那么就将其索引化，索引化的方法经下面的试验得到的结论为:特征的取值异数大于等于maxC的不作变化，小于的
    #如值为0，则编号就为0，其他的按照频数越高序号越高的原则,没有取值为0的，就按照频数越高序号越高的原则排序。这跟stringindexer反过来了
    dataframe = spark.createDataFrame(
        [  (Vectors.dense(2.0, 3.0, 1.0),),
           (Vectors.dense(2.0, 3.0, 1.0),),
           (Vectors.dense(6.0, 0.0, 1.0),),
           (Vectors.dense(6.0, 0.0, 2.0),),
           (Vectors.dense(6.0, 0.0, 2.0),),
           (Vectors.dense(6.0, 0.0, 2.0),),
           (Vectors.dense(6.0, 0.0, 2.0),),
           (Vectors.dense(6.0, 0.0, 3.0),),
           (Vectors.dense(6.0, 0.0, 3.0),),
           (Vectors.dense(6.0, 0.0, 3.0),),
           (Vectors.dense(6.0, 0.0, 3.0),),
           (Vectors.dense(6.0, 0.0, 3.0),)],
        ["features"])
    vector_indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=3)
    vector_indexer_model = vector_indexer.fit(dataframe)#这东西也是个评估器
    result = vector_indexer_model.transform(dataframe)
    result.show()

    # VectorAssembler的作用是将多个列合并为一个向量列
    dataframe = spark.createDataFrame([(0, "a"), (1, "b"), (2, "c"), (3, "a"), (4, "a"), (5, "c"), (6, "d"), (7, "e")],
                                      ["id", "category"])
    indexer = StringIndexer(inputCol="category", outputCol="categoryIndexed")
    indexer_model = indexer.fit(dataframe)
    indexed = indexer_model.transform(dataframe)
    vector_assembler = VectorAssembler(inputCols=["id", "categoryIndexed"], outputCol="vectorTest")
    vector_result = vector_assembler.transform(indexed)
    vector_result.show()
    print("转为RDD")
    vector_rdd = vector_result.rdd
    print(vector_rdd.collect())  # 通过打印rdd可以看到经过VectorAssembler操作的结果也是DensVector类型,和vectorindexer操作的source列一样

indexer_and_assembler()