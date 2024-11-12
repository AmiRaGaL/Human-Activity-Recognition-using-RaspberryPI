# Human Activity Recognition with Fall Detection

This project implements a comprehensive **Human Activity Recognition (HAR)** system using a Raspberry Pi camera with real-time **fall detection** and **notification** capabilities. The project combines **Convolutional Neural Networks (CNNs)**, **Long Short-Term Memory (LSTM)** networks, and **Deep Neural Networks (DNNs)** using RGB and Pose-based algorithms to enhance activity recognition. Integrating with AWS IoT Core and SNS, the system can notify predefined contacts in case of a detected fall.

## Project Structure

| Directory/File       | Description |
|----------------------|-------------|
| **AWS_Connect**      | Code to trigger custom notifications through AWS. |
| **RGB_Pose_Prediction** | Predicts the video class based on RGB and Pose algorithm models. |
| **ConvLSTM_Model**   | Implementation, training, and evaluation of ConvLSTM approach. |
| **LRCN_Model**       | Implementation, training, and evaluation of the LRCN approach. |
| **data_collection**  | Extracts pose model features, capturing video frames and saving them as `.npy` files. |
| **data_training**    | Loads `.npy` files to train the pose model. |
| **Fall_Pred_Notif**  | Prediction and notification trigger for fall events. |
| **inference**        | Simple model prediction code. |
| **Pose**             | Mediapipe-based human pose tracking. |
| **TfliteConverter**  | Converts `.h5` models to `.tflite` format for IoT devices. |
| **Video_Capture**    | Captures video from a camera and stores it locally. |
| **Video_Resize**     | Preprocesses custom videos by adjusting resolution. |

## System Overview

The HAR system uses a **Joint RGB-Pose-based** algorithm to capture spatial and temporal features, enhancing activity recognition accuracy. The core components include:

1. **Data Collection and Preprocessing**: Utilizes datasets (HMDB51, UCF50, KTH) and custom data for fall detection. Videos are standardized to 640x480 resolution, and essential joints (shoulder, knee) are maintained in Pose-based videos.
   
2. **Machine Learning Models**:
   - **ConvLSTM**: Recognizes RGB-based activities using spatial and temporal features.
   - **LRCN**: Combines CNN and LSTM layers for efficient video classification.
   - **Pose-based DNN**: Uses Mediapipe-extracted pose key points to classify actions.

3. **Fall Detection and Notification**: Integrates AWS IoT Core and SNS for real-time fall detection and notification via email alerts.

## Hardware and Software Requirements

- **Hardware**: Raspberry Pi 4 Model B, Camera Module, Micro SD Card, Power Supply
- **Software**: Python, PyCharm/Jupyter Notebook, AWS IoT Core, SNS, Mediapipe, OpenCV

## Setup and Installation

1. **Data Collection**: Follow instructions in `data_collection` to capture frames and save as `.npy` files.
2. **Model Training**: Use `ConvLSTM_Model` and `LRCN_Model` for RGB-based training, and `Pose` for Pose-based training.
3. **AWS Notification**: Set up AWS IoT Core and SNS to enable email notifications.

## Results

Our LRCN model achieved over 90% accuracy on validation data, and Pose-based DNN showed above 98% accuracy in real-time scenarios.

## Future Work

- **User-Defined Anomalies**: Enable customization of anomaly detection.
- **Enhanced User Interface**: Provide web/mobile access for configuration and monitoring.
- **Privacy & Security**: Address privacy concerns in public applications.

## Acknowledgments

Special thanks to **OpenCV**, **Mediapipe**, **AWS Documentation**, and **Bleed AI Academy**.
"# Human-Activity-Recognition-using-RaspberryPI" 
