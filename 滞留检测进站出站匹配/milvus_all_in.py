from pymilvus import connections,utility,Collection, DataType, FieldSchema, CollectionSchema

#单机、单
class MilvusAllIn:
    def __init__(self,ip,port,alias):
        connections.connect(
            alias=alias,
            host=ip,
            port=port
        )

