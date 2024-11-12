import cv2


video_name = input("Enter the name of the video : ")
# Create a VideoCapture object
cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera, you can change it if you have multiple cameras

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('Captures/'+video_name+'.avi', fourcc, 20.0, (640, 480))  # Change 'output.avi' to your desired output file name

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Write the frame to the output video file
        out.write(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()