from pymilvus import connections,utility,Collection, DataType, FieldSchema, CollectionSchema

class Person:
    def __init__(self,collection_name):
        person_id = FieldSchema(
            name="person_id",
            dtype=DataType.INT64,
            is_primary=True,
        )
        stay_time = FieldSchema(
            name="stay_time",
            dtype=DataType.INT64,
            description='the time'
        )
#单机、单partition
class MilvusAllIn:
    def __init__(self,ip,port,alias):
        connections.connect(
            alias=alias,
            host=ip,
            port=port
        )
        #总人数。项目运行时获取保存的总人数。每当插入时增加，删除时减少
        self.total_persons = 0
        #每新插入一定数目的数据就重新建立索引
        self.new_insert = 0


