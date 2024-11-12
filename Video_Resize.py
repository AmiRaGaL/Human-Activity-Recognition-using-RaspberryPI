import os

import cv2
from moviepy.video.io.VideoFileClip import VideoFileClip


def resize_video(video_path,class_dir):
    # Input and output file paths
    #input_video_path = 'fall_pp/fall-26.mp4'
    os.makedirs(class_dir+'_resized', exist_ok=True)
    input_video_path = os.path.join(class_dir, video_path)
    output_video_path = os.path.join(class_dir+'_resized',video_path)
    target_width = 640
    target_height = 480
    # Load the input video clip
    resized_clip = VideoFileClip(input_video_path, target_resolution=(target_height, target_width))

    # Resize video without changing the duration
    #resized_clip = clip.resize((target_width, target_height))

    # Write the resized video to the output file while preserving the original audio
    resized_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")

    # # Target dimensions
    # target_width = 640
    # target_height = 480
    #
    # # Open the input video file
    # cap = cv2.VideoCapture(input_video_path)
    #
    # # Get the video's frames per second (fps) and size
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #
    # # Create VideoWriter object to save the resized video
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    # out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))

    # Read and resize each frame of the video
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #
    #     # Resize the frame to the target dimensions
    #     resized_frame = cv2.resize(frame, (target_width, target_height))
    #
    # # Write the resized frame to the output video file
    # out.write(resized_frame)
    #
    # # Release the video capture and writer objects
    # cap.release()
    # out.release()
    #
    # # Close any OpenCV windows
    # cv2.destroyAllWindows()

# #class_dir = input("Enter the path of the video : ")
class_dir = 'fall_pp'
for video_file in os.listdir(class_dir):
    print(video_file)
    #video_path = os.path.join(class_dir, video_file)
    #print(video_path)
    resize_video(video_file,class_dir)

resize_video('fall_pp/fall-11.mp4','fall_pp')