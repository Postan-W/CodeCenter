from pymilvus import connections,utility,Collection, DataType, FieldSchema, CollectionSchema
connector = connections.connect(
  alias="default",
  host='10.128.12.15',
  port='19530'
)
