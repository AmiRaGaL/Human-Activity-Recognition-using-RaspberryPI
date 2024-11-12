import time
import paho.mqtt.client as mqtt
import ssl
import json
import _thread
#import RPi.GPIO as GPIO

# GPIO.setmode(GPIO.BCM)
# GPIO.setup(21, GPIO.OUT)
def Push_Notification(topic, message):

    def on_connect(client, userdata, flags, rc):
        print("Connected with result code "+str(rc))


    client = mqtt.Client()
    client.on_connect = on_connect
    client.tls_set(ca_certs='./AmazonRootCA1.pem', certfile='./0c933f93393b319a206e9f4b25dd5f7f0861021bb63ea166b9d2e5f9f5617bfb-certificate.pem.crt', keyfile='./0c933f93393b319a206e9f4b25dd5f7f0861021bb63ea166b9d2e5f9f5617bfb-private.pem.key', tls_version=ssl.PROTOCOL_SSLv23)
    client.tls_insecure_set(True)
    client.connect("a2u7bsn4q3kxrq-ats.iot.us-east-2.amazonaws.com", 8883, 60) #Taken from REST API endpoint - Use your own.
    time.sleep(5)

    def intrusionDetector(Dummy):
        print("Pushing Notification : "+message)
        client.publish(topic, payload=message, qos=0, retain=False)
        return 0
    client.loop_start()
    intrusionDetector(0)
    client.loop_stop()

if __name__ == "__main__":
    Push_Notification("device/data", "AWS Notification Test Run")


