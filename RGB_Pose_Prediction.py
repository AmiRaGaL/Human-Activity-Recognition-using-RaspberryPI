import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import tensorflow as tf
import cv2
import numpy as np

# Define the image dimensions
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

# Define your list of classes
CLASSES_LIST = ["fall_floor", "No_fall"]

def predict_live_action(SEQUENCE_LENGTH):
    '''
    This function will perform live action recognition prediction using the LRCN model.
    Args:
    SEQUENCE_LENGTH: The fixed number of frames of a video that can be passed to the model as one sequence.
    '''

    # Load the pre-trained model
    LRCN_model = tf.keras.models.load_model('LRCN_model___Date_Time_2023_12_06__18_15_36.h5')

    # Initialize the VideoCapture object for live video capture (camera index 0 usually corresponds to the default webcam).
    video_capture = cv2.VideoCapture(0)

    # Initialize a variable to store the predicted action being performed in the video.
    predicted_class_name = ''

    while True:
        # Read a frame from the video capture
        success, frame = video_capture.read()

        # Resize the frame to the fixed dimensions
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Normalize the resized frame
        normalized_frame = resized_frame / 255

        # Display the frame
        cv2.imshow("Live Video", frame)

        # Append the normalized frame to the frames list
        frames_list = [normalized_frame] * SEQUENCE_LENGTH

        # Convert the frames list to a numpy array and add a batch dimension
        frames_array = np.expand_dims(frames_list, axis=0)

        # Predict the action using the model
        predicted_labels_probabilities = LRCN_model.predict(frames_array)[0]

        # Get the index of the class with the highest probability
        predicted_label = np.argmax(predicted_labels_probabilities)

        # Get the class name using the retrieved index
        predicted_class_name = CLASSES_LIST[predicted_label]

        # Display the predicted action along with the prediction confidence
        #print(f'Action Predicted: {predicted_class_name}\nConfidence: {predicted_labels_probabilities[predicted_label]}')
        cv2.putText(frame, f'{predicted_class_name} : accuracy -{predicted_labels_probabilities[predicted_label]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("window", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the VideoCapture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
    return predicted_class_name, predicted_labels_probabilities[predicted_label]
def inFrame(lst):
    if lst[27].visibility > 0.6 and lst[26].visibility > 0.6 and lst[15].visibility > 0.6 and lst[16].visibility > 0.6:
        return True
    return False


model = load_model("fall_model.h5")
label = np.load("Data/labels.npy")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("KTH/handclapping/person01_handclapping_d3_uncomp.avi")


while True:
    lst = []
    lstm_class,lstm_accuracy = 0,0#predict_live_action(20)
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
        p_acc =p[0][np.argmax(p)]
        # print(f"Predicted Pose: {pred} accuracy: {p[0][np.argmax(p)] * 100} %")

        if p_acc > 0.75 and p_acc > lstm_accuracy:
            cv2.putText(window, pred, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
            print(f"Predicted Pose: {pred} accuracy: {p[0][np.argmax(p)] * 100} %")

        elif lstm_accuracy > 0.75 and lstm_accuracy > p_acc:
            cv2.putText(window, lstm_class, (180, 180), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
            print(f"Predicted Pose: {lstm_class} accuracy: {p[0][np.argmax(p)] * 100} %")

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
        break

