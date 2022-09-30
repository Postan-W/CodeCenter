from pymilvus import utility,connections,Collection,Partition
from milvus_basic import connect_the_milvus
connect_the_milvus()

def get_all_collections():
    return utility.list_collections()

def delete_the_collection(name):
    utility.drop_collection(name)

# delete_the_collection("timestamp")
# print(get_all_collections())
"""
A collection alias is globally unique, hence you cannot assign the same alias 
to different collections. However, you can assign multiple aliases to one collection.
"""
def create_alias():
    utility.create_alias(
        collection_name="test_time_travel",
        alias="publication"
    )



def drop_alias():
    utility.drop_alias(alias="publication")

#"publication"本来是test_time_travel的alias，现在改为book的别称，这就是alter_alias的作用
def alter_alias():
    utility.alter_alias(
        collection_name="book",
        alias="publication"
    )

"""
This topic describes how to load the collection to memory before a search or a query. 
All search and query operations within Milvus are executed in memory.
In current release, volume of the data to load must be under 90% of the total memory resources of 
all query nodes to reserve memory resources for execution engine.
"""
def load_in_memory():
    collection = Collection("book")
    collection.load()
    # collection.release()

"""
Milvus allows you to divide the bulk of vector data into a small number of partitions. Search and other
operations can then be limited to one partition to improve the performance.A collection consists of one or
more partitions. While creating a new collection, Milvus creates a default partition _default. See 
Glossary - Partition for more information.
"""
def create_partition():
    collection = Collection("book")
    collection.create_partition("first")


def check_partition():
    collection = Collection("book")
    print(collection.has_partition("first"))
    print(collection.partitions)

def drop_partition():
    collection = Collection("book")
    collection.drop_partition("first")
    print(collection.partitions)

#Optionally Loading partitions instead of the whole collection to memory can significantly reduce the memory usage
def load_partition():
    collection = Collection("book")
    # collection.load(["partition_name"])
    #或者是生成一个partition对象
    # partition = Partition("partition_name")
    # partition.load()
    #partition.release()

