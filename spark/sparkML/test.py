from base_config import *
from pyspark.ml.feature import OneHotEncoder,VectorIndexer,StringIndexer,IndexToString,VectorAssembler
data = spark.createDataFrame([['c','d','e'],
                                               ['c','d','f'],
                                               ['g','h','c'],
                                               ['m','k','v'],
                                               ['c','m','t'],
                                               ['g','s','x']],['first','second','third'])
indexer = StringIndexer(inputCol='first',outputCol='firstIndex')
data = indexer.fit(data).transform(data)
data.show()#0,0,1,2,0,1
