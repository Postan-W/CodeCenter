from paho.mqtt import client as mqtt_client
import sys
from main import Detector
from utils import *
import json
import random

host = '10.0.0.31'
port = 1883
topic_origin_image = "/iot/v3/bkimg/U/P/DIMG2"
client_sub_id = "bk_img_input2_receiver"
topic_processed_image = "/iot/v3/bkcm/U/P/CM2"
client_pub_id = "bk_cm_input2"
count = 0

def connect_mqtt(client_id):
    #连接mqtt服务器
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            if client_id.startswith("bk_img"):
                print("接收者:{}连上mqtt".format(client_id))
            elif client_id.startswith("bk_cm"):
                print("发送者{}连上mqtt".format(client_id))
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.on_connect = on_connect
    # broker = 'broker.emqx.io'
    # port = 1883
    # client.connect(broker, port)
    client.connect(host=host, port=port)
    return client

class Processor:
    def __init__(self,client_sub,client_pub):
        self.sub = connect_mqtt(client_sub)
        self.pub = connect_mqtt(client_pub)
        self.detect = Detector()

    def subscriber(self):
        def on_message(client, userdata, msg):
            messa = msg.payload.decode()
            messa = eval(messa)
            try:
                img_code, caremaID = messa.get('data'), messa.get("caremaID")
                is_bs64 = check_bs64(img_code)
                if is_bs64:
                    img = basedata2np(img_code)
                    image, image_alert, count = self.detect.detect(img)
                    image = img2bs64(image)
                    image_alert = img2bs64(image_alert)
                    res = {"image": image, "image_alert": image_alert, "nums": count, "caremaID": caremaID}
                    res = json.dumps(res)
                    result = self.pub.publish(topic_processed_image, res)
                    status = result[0]
                    if status == 0:
                        count += 1
                        if count > 100000:
                            count = 0
                        print("处理过的图片{}发送成功".format(count))
                    else:
                        print("处理过的图片发送失败")
            except Exception as e:
                print("处理出错:{}".format(e))

        self.sub.subscribe(topic_origin_image)
        self.sub.on_message = on_message

if __name__ == "__main__":
    processor = Processor(client_sub_id,client_pub_id)
    processor.subscriber()
    processor.sub.loop_forever()








