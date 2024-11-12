import _thread

import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
#from AWS_Connect import Push_Notification
import time
import paho.mqtt.client as mqtt
import ssl

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))


client = mqtt.Client()
client.on_connect = on_connect
client.tls_set(ca_certs='./AmazonRootCA1.pem', certfile='./0c933f93393b319a206e9f4b25dd5f7f0861021bb63ea166b9d2e5f9f5617bfb-certificate.pem.crt', keyfile='./0c933f93393b319a206e9f4b25dd5f7f0861021bb63ea166b9d2e5f9f5617bfb-private.pem.key', tls_version=ssl.PROTOCOL_SSLv23)
client.tls_insecure_set(True)
client.connect("a2u7bsn4q3kxrq-ats.iot.us-east-2.amazonaws.com", 8883, 60) #Taken from REST API endpoint - Use your own.


def inFrame(lst):
    if lst[27].visibility > 0.6 and lst[26].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False
def Push_Notification(topic, message):
    print("Pushing Notification : "+message)
    client.publish("device/data", payload=message, qos=0, retain=False)

model = load_model("fall_model.h5")
label = np.load("labels_fall.npy")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils
fall_frames = 5

def Capture(x):
    counter = 0
    cap = cv2.VideoCapture(0)
    while True:
        lst = []

        _, frm = cap.read()

        window = np.zeros((940, 940, 3), dtype="uint8")

        frm = cv2.flip(frm, 1)

        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        frm = cv2.blur(frm, (4, 4))
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)

            p = model.predict(lst)
            pred = label[np.argmax(p)]
            print(f"Predicted Pose: {pred} accuracy: {p[0][np.argmax(p)] * 100} %")

            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(window, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
                if pred == "fall_floor":
                    print("Fall Detected")
                    counter += 1
                    if counter == fall_frames:
                        Push_Notification('data/device', 'Fall Detected')
                        counter = 0



            else:
                cv2.putText(window, "Unknown Pose", (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)

        else:
            cv2.putText(frm, "Make Sure Full body visible", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                               connection_drawing_spec=drawing.DrawingSpec(color=(255, 255, 255), thickness=6),
                               landmark_drawing_spec=drawing.DrawingSpec(color=(0, 0, 255), circle_radius=3, thickness=3))

        window[420:900, 170:810, :] = cv2.resize(frm, (640, 480))

        cv2.imshow("window", window)

        if cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            cap.release()
            client.disconnect()
            break

_thread.start_new_thread(Capture, ("Create intrusion Thread",))

client.loop_forever()