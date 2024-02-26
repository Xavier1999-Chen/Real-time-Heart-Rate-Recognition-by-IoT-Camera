import paho.mqtt.client as mqtt
import base64
import binascii
import numpy as np
import cv2

# # 定义回调函数，当接收到消息时调用
# def on_message(client, userdata, msg):
#     # 获取十六进制编码的图片数据
#     img_data_hex = msg.payload #.decode()
#     # 解码图片数据
#     try:
#         img_data = base64.b64decode(img_data_hex)
#         # img_data = binascii.unhexlify(img_data_hex)
#         # 将图片保存到文件
#         with open("image.png", "wb") as f:
#             f.write(img_data)
#         print(len(img_data))
#         print("Received image!")
#         # nparr = np.frombuffer(img_data, np.uint8)
#         # img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         # cv2.imshow("Image", img_cv)
#         # cv2.waitKey(0)
        
#     except binascii.Error as e:
#         print("Error decoding image:", e)

def on_message(client, userdata, message):
    print(f"Received message: {message.payload.decode()+':81/stream'}")

# 输入MQTT的用户名和密码
mqtt_username = "aiot20"
mqtt_password = "12345610"

# 创建MQTT客户端实例
client = mqtt.Client()
# 设置用户名和密码
client.username_pw_set(mqtt_username, mqtt_password)
# 设置回调函数
client.on_message = on_message
# 连接MQTT服务器
client.connect("p224517a.emqx.cloud", 1883, 60)
# 订阅MQTT topic
client.subscribe("aiot/esp32/test")
# 开始循环，等待消息
client.loop_forever()