以下内容为简单实践，请仔细参考官网教程https://milvus.io/docs/v2.0.x/manage_connection.md。

系统性总结:https://blog.csdn.net/scgaliguodong123_/article/details/123281018

## 1.启动服务

使用docker-compose启动服务(当然,也可以手动启相关容器)。

**安装docker-compose**：

下载到/usr/local/bin下：

curl -L https://github.com/docker/compose/releases/download/1.23.1/docker-compose-`uname -s`-`uname -m` > /usr/local/bin/docker-compose

one more下载地址:

https://gitee.com/boring-yingjie/docker-compose

然后增加docker-compose的可执行权限即可。

**下载milvus项目的compose文件**：

wget https://github.com/milvus-io/milvus/releases/download/v2.0.2/milvus-standalone-docker-compose.yml -O docker-compose.yml

内容如下：

```
version: '3.5'

services:
  etcd:
    container_name: milvus-etcd
    image: quay.io/coreos/etcd:v3.5.0
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

  minio:
    container_name: milvus-minio
    image: minio/minio:RELEASE.2020-12-03T00-03-10Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/minio:/minio_data
    command: minio server /minio_data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  standalone:
    container_name: milvus-standalone
    image: milvusdb/milvus:v2.0.2
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ${DOCKER_VOLUME_DIRECTORY:-.}/volumes/milvus:/var/lib/milvus
    ports:
      - "19530:19530"
    depends_on:
      - "etcd"
      - "minio"

networks:
  default:
    name: milvus

```

在compose中把其所在的目录作为作为了各个容器挂载宿主机的目录的前缀，运行后可以看见"当前目录/volumes"的产生。

**启动**：

docker-compose up -d

![image-20220927161238229](mdimages/image-20220927161238229.png)

**查看：**

docker-compose ps

![image-20220927161258472](mdimages/image-20220927161258472.png)

**停止**:

docker-compose down

停止之后建议把挂载的volumes目录给删了。

## 2.使用(pymilvus version:2.1.2)

### 2.1连接milvus服务

```
from pymilvus import connections
connections.connect(
  alias="default", 
  host='localhost', 
  port='19530'
)
```

![image-20220927175138898](mdimages/image-20220927175138898.png)

断开连接：

para:Alias of the Milvus server to disconnect from.

connections.disconnect(alias="default")

### 2.2创建collection

collection可以对照结构化数据库，理解为数据表，创建collection需要定义CollectionSchema，CollectionSchema中包含n个FieldSchema，可以理解为字段，最后collection需要一个name，表名。

```
from pymilvus import CollectionSchema, FieldSchema, DataType
#主filed,必须是INT64
book_id = FieldSchema(
  name="book_id", 
  dtype=DataType.INT64, 
  is_primary=True, 
)
word_count = FieldSchema(
  name="word_count", 
  dtype=DataType.INT64,  
)
#这里的向量是2维。维度范围[1, 32,768]
book_intro = FieldSchema(
  name="book_intro", 
  dtype=DataType.FLOAT_VECTOR, 
  dim=2
)
#定义利用上面的field定义CollectionSchema
schema = CollectionSchema(
  fields=[book_id, word_count, book_intro], 
  description="Test book search"
)
collection_name = "book"
#创建collection
collection = Collection(
    name=collection_name, 
    schema=schema, 
    using='default', 
    shards_num=2,
    consistency_level="Strong"
    )
```

### 2.3插入数据

The following example inserts 2,000 rows of randomly generated data as the example data (Milvus CLI example uses a pre-built, remote CSV file containing similar data). Real applications will likely use much higher dimensional vectors than the example. You can prepare your own data to replace the example.

```
data = [
  [i for i in range(2000)],
  [i for i in range(10000, 12000)],
  [[random.random() for _ in range(2)] for _ in range(2000)],
]
#获取一个已经存在的collection。当然，这里的代码和上面创建的代码写在一起，可以不执行该句
collection = Collection("book")
mr = collection.insert(data)
```

insert的参数：

![image-20220927180922768](mdimages/image-20220927180922768.png)

A collection consists of one or more partitions. While creating a new collection, Milvus creates a default partition `_default`.

### 2.4为vector建立索引

Vector indexes are an organizational unit of metadata used to accelerate [vector similarity search](https://milvus.io/docs/v2.0.x/search.md). Without index built on vectors, Milvus will perform a brute-force search by default.

```
#定义索引
index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}
#只能为vector field建立索引
collection.create_index(
  field_name="book_intro", 
  index_params=index_params
)
```

### 2.5搜索

All search and query operations within Milvus are executed in memory. Load the collection to memory before conducting a vector similarity search.

```
collection = Collection("book")# Get an existing collection.
collection.load()
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
#可以利用expr过滤结果。results是长度为10的列表
results = collection.search(
	data=[[0.1, 0.2]], 
	anns_field="book_intro", 
	param=search_params, 
	limit=10, 
	expr=None,
	consistency_level="Strong"
)
collection.release()#释放内存

```

