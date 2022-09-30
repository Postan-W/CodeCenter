from pymilvus import connections,utility,Collection, DataType, FieldSchema, CollectionSchema
import random
import datetime

def connect_the_milvus():
    connections.connect(
        alias="default",
        host='10.128.12.15',
        port='19530'
    )

def disconnect_milvus():
    connections.disconnect(alias="default")

def define_collection():
    book_id = FieldSchema(
        name="book_id",
        dtype=DataType.INT64,
        is_primary=True,
    )
    word_count = FieldSchema(
        name="word_count",
        dtype=DataType.INT64,
        description="the count of words in the book"
    )
    # 这里的向量是2维。[1, 32,768]
    book_intro = FieldSchema(
        name="book_intro",
        dtype=DataType.FLOAT_VECTOR,
        dim=2
    )
    # 定义利用上面的field定义CollectionSchema
    schema = CollectionSchema(
        fields=[book_id, word_count, book_intro],
        description="Test book search"
    )
    collection_name = "book"
    # 创建collection
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2,
        consistency_level="Strong"
    )
    return collection


def insert_data():
    connect_the_milvus()
    # 可以通过名称获取已经创建的collection
    collection = Collection("book")
    data = [
        [i for i in range(2000)],
        [i for i in range(10000, 12000)],
        [[random.random() for _ in range(2)] for _ in range(2000)],
    ]
    mr = collection.insert(data)#参数partition_name可以指定往哪个partition插入数据
    disconnect_milvus()

def build_index():
    connect_the_milvus()
    # 可以通过名称获取已经创建的collection
    collection = Collection("book")
    # 定义索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    # 只能为vector field建立索引
    collection.create_index(
        field_name="book_intro",
        index_params=index_params
    )
    # collection.drop_index()
    disconnect_milvus()

#相似度匹配
def similarity_search():
    connect_the_milvus()
    collection = Collection("book")  #Get an existing collection.
    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    # 匹配一个向量。可以利用expr过滤结果。results是长度为10的列表
    results = collection.search(
        data=[[0.1, 0.2]],
        anns_field="book_intro",
        param=search_params,
        limit=10,
        expr=None,
        consistency_level="Strong"
    )
    print(type(results[0]))
    print(list(results[0]))
    collection.release()  #释放内存
    disconnect_milvus()

#查询
def search_certain():
    connect_the_milvus()
    # 可以通过名称获取已经创建的collection
    collection = Collection("book")
    collection.load()
    res = collection.query(
        expr="book_id in [2,4,6,8]",
        output_fields=["book_id","word_count","book_intro"],
        consistency_level="Strong"
    )
    res = list(res)
    print(res)
    disconnect_milvus()

def hybrid_search():
    """
    Suppose you want to search for certain books based on their vectorized introductions,
    but you only want those within a specific range of word count. You can then specify
    the boolean expression to filter the word_count field in the search parameters.
    Milvus will search for similar vectors only among those entities that match the expression.
    :return:
    """
    connect_the_milvus()
    collection = Collection("book")  # Get an existing collection.
    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    # 匹配一个向量。可以利用expr过滤结果。results是长度为10的列表
    results = collection.search(
        data=[[0.1, 0.2]],
        anns_field="book_intro",
        param=search_params,
        limit=10,
        expr="word_count <= 11000",
        consistency_level="Strong"
    )
    print(results[0])
    disconnect_milvus()

#search with timestamp。Milvus maintains a timeline for all data insert and delete operations
def search_with_timestamp():
    """
    下面演示了三次插入数据，序号是0-30，搜索的时候按照第二次的timestamp进行搜索，只返回了0-19的结果，说明timestamp的作用是搜索该时间点之前的
    :return:
    """
    connect_the_milvus()
    collection_name = "test_time_travel"
    schema = CollectionSchema([
        FieldSchema("pk", DataType.INT64, is_primary=True),
        FieldSchema("example_field", dtype=DataType.FLOAT_VECTOR, dim=2)
    ])
    collection = Collection(collection_name, schema)
    data = [
        [i for i in range(10)],
        [[random.random() for _ in range(2)] for _ in range(10)],
    ]
    # batch1 = collection.insert(data)#(insert count: 10, delete count: 0, upsert count: 0, timestamp: 436298566327861250, success count: 10, err count: 0)
    data = [
        [i for i in range(10, 20)],
        [[random.random() for _ in range(2)] for _ in range(10)],
    ]
    batch2 = collection.insert(data)
    data = [
        [i for i in range(20, 30)],
        [[random.random() for _ in range(2)] for _ in range(10)],
    ]
    batch3 = collection.insert(data)
    collection.load()
    search_param = {
        "data": [[1.0, 1.0]],
        "anns_field": "example_field",
        "param": {"metric_type": "L2"},
        "limit": 100,
        "travel_timestamp": batch2.timestamp,
    }
    res = collection.search(**search_param)
    print(res[0].ids)

