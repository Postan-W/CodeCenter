# -*- coding: utf-8 -*-
# @Time    : 2022/5/30 15:11
# @Author  : Chuck

import json
from paho.mqtt import client as mqtt_client
from main import Detector
from utils import *

class MQTT(object):
    def __init__(self, broker, port, topic_sub, topic_pub, client_id_sub, client_id_pub):
        self.broker = broker
        self.port = port
        self.topic_sub = topic_sub
        self.topic_pub = topic_pub
        self.client_sub = self.connect_mqtt(client_id_sub)
        self.client_pub = self.connect_mqtt(client_id_pub)
        self.detect = Detector

    def connect_mqtt(self, client_id):
        '''连接mqtt代理服务器'''

        def on_connect(client, userdata, flags, rc):
            '''连接回调函数'''
            # 响应状态码为0表示连接成功
            if rc == 0:
                print("Connected to MQTT OK!")
            else:
                print("Failed to connect, return code %d\n", rc)

        # 连接mqtt代理服务器，并获取连接引用
        client = mqtt_client.Client(client_id)
        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    def subscribe(self):
        '''订阅主题并接收消息'''

        def on_message(client, userdata, msg):
            '''订阅消息回调函数'''
            messa = msg.payload.decode()
            topic = msg.topic
            messa = eval(messa)
            try:
                img_code, caremaID = messa.get('img'), messa.get("caremaID")
                is_bs64 = check_bs64(img_code)
                if is_bs64:
                    img = basedata2np(img_code)
                    image, image_alert, count = self.detect.detect(img)
                    image = img2bs64(image)
                    image_alert = img2bs64(image_alert)
                    res = {"image": image, "image_alert": image_alert, "nums": count, "caremaID": caremaID}
                    res = json.dumps(res)
                    result = self.client_pub.publish(self.topic_pub, res)
                    status = result[0]
                    if status == 0:
                        print(f"Send `{res}` to topic `{self.topic_pub}`")
                        logger.info(f"Send `{res}` to topic `{self.topic_pub}`")
                    else:
                        print(f"Failed to send message to topic {self.topic_pub}")
                        logger.info(f"Failed to send message to topic {self.topic_pub}")
            except Exception as e:
                logger.info("检测失败", e)

        # 订阅指定消息主题
        self.client_sub.subscribe(self.topic_sub)
        self.client_sub.on_message = on_message

    def run(self):
        # 运行订阅者
        self.subscribe()
        #  运行一个线程来自动调用loop()处理网络事件, 阻塞模式
        self.client_sub.loop_forever()


if __name__ == '__main__':
    broker = '10.20.190.41'
    port = 1883
    topic_sub = "/iot/v3/bkimg/U/P/DIMG1"
    topic_pub = "/iot/v3/bkcm/U/P/CM1"
    client_id_sub = "bk_img_input1"
    client_id_pub = "bk_cm_input1"

    M = MQTT(broker, port, topic_sub, topic_pub, client_id_sub, client_id_pub)
    M.run()
