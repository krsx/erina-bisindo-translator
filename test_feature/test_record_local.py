import cv2
import time

from random import random

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f'output{random()}.avi', fourcc, 20.0, (640, 480))

# Open default camera
cap = cv2.VideoCapture(0)

# Variables for FPS calculation
start_time = time.time()
num_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Write the frame
    out.write(frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Increment frame count
    num_frames += 1

    # Calculate FPS
    elapsed_time = time.time() - start_time
    fps = num_frames / elapsed_time
    cv2.putText(frame, f"FPS: {round(fps, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()