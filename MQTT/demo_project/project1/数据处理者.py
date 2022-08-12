from paho.mqtt import client as mqtt_client
import sys

topic_origin_image = "origin_image"
client_sub_id = "client2"
topic_processed_image = "processed_image"
client_pub_id = "client3"

def connect_mqtt(client_id):
    #连接mqtt服务器
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("{} Connected to MQTT Broker!".format(client_id))
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    # broker = 'broker.emqx.io'
    # port = 1883
    # client.connect(broker, port)
    client.connect(host='127.0.0.1', port=1883)
    return client

class Processor:
    def __init__(self,client_sub,client_pub):
        self.sub = connect_mqtt(client_sub)
        self.pub = connect_mqtt(client_pub)

    def subscriber(self):
        def on_message(client, userdata, msg):
            data = msg.payload.decode()
            print('订阅【{}】的消息为：{}'.format(msg.topic, round(sys.getsizeof(data)/1024,1)))
            result = self.pub.publish(topic_processed_image, data)
            status = result[0]
            if status == 0:
                print("处理过的图片发送成功")
            else:
                print("处理过的图片发送失败")

        self.sub.subscribe(topic_origin_image)
        self.sub.on_message = on_message

if __name__ == "__main__":
    processor = Processor(client_sub_id,client_pub_id)
    processor.subscriber()
    processor.pub.loop_start()
    processor.sub.loop_forever()
    print("测试")





