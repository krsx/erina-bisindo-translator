import streamlit as st
import utils.settings as settings
from streamlit_webrtc import VideoTransformerBase
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import math
from mediapipe.framework.formats import landmark_pb2
import av
import time
import asyncio
import os
import threading
from gtts import gTTS
import tempfile
import sounddevice as sd
import soundfile as sf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

lock = threading.Lock()
shared_fps = 0

# FUNCTION FOR MEDIAPIPE

def media_pipe_detection(image, model):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    return image, results   

def draw_land_marks(image, results):
   custom_pose_connections = list(mp_pose.POSE_CONNECTIONS)

   excluded_landmarks = [
        PoseLandmark.NOSE,
        PoseLandmark.LEFT_EYE_INNER,
        PoseLandmark.LEFT_EYE,
        PoseLandmark.LEFT_EYE_OUTER,
        PoseLandmark.RIGHT_EYE_INNER,
        PoseLandmark.RIGHT_EYE,
        PoseLandmark.RIGHT_EYE_OUTER,
        PoseLandmark.LEFT_EAR,
        PoseLandmark.RIGHT_EAR,
        PoseLandmark.MOUTH_LEFT,
        PoseLandmark.MOUTH_RIGHT,
        PoseLandmark.LEFT_HIP,
        PoseLandmark.RIGHT_HIP,
        PoseLandmark.LEFT_KNEE,
        PoseLandmark.RIGHT_KNEE,
        PoseLandmark.LEFT_ANKLE,
        PoseLandmark.RIGHT_ANKLE,
        PoseLandmark.LEFT_HEEL,
        PoseLandmark.RIGHT_HEEL,
        PoseLandmark.LEFT_FOOT_INDEX,
        PoseLandmark.RIGHT_FOOT_INDEX
    ]

   for landmark in excluded_landmarks:
        custom_pose_connections = [
            connection_tuple for connection_tuple in custom_pose_connections if landmark.value not in connection_tuple]

   mp_drawing.draw_landmarks(
        image, results.pose_landmarks, connections=custom_pose_connections)
   mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
   mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def mediapipe_callback(frame):
    global prev_frame_time
    # global shared_fps

    new_frame_time = time.time()

    img = frame.to_ndarray(format="bgr24")

    img, results = media_pipe_detection(img, holistic_model)

    draw_land_marks(img, results)

    # fps = 1 / (new_frame_time - prev_frame_time)
    # prev_frame_time = new_frame_time

    # Log FPS or update an external metric
    # print(f"Calculated FPS: {fps:.2f}") 
    # if 'fps' not in st.session_state:
    #     st.session_state['fps'] = 0

    # st.session_state['fps'] = fps

    # if 'fps' in st.session_state:
    #     print(f"Calculated FPS: {fps:.2f}") 

    # with lock:
    #     shared_fps = fps

    # Convert the processed image back to a VideoFrame
    new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")

    return new_frame

# # FUNCTION FOR LSTM
# def load_lstm_model():
#     model = Sequential()

#     model.add(TimeDistributed(Dense(units=128, activation='tanh'), input_shape=(30, 108)))
#     model.add(LSTM(128, return_sequences=True, activation='tanh'))
#     model.add(Dropout(0.5))
#     model.add(LSTM(64, return_sequences=False, activation='tanh'))
#     model.add(Dropout(0.5))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(settings.MODEL_ACTIONS.shape[0], activation='softmax'))

#     model.load_weights('model-bimbingan5v3.h5')

#     return model 

# def prob_viz(res, actions, input_frame, frame_height=480, frame_width=640, opacity=0.4):
#     colors = [
#         (245, 117, 16),  # Orange
#         (117, 245, 16),  # Lime Green
#         (16, 117, 245),  # Bright Blue
#         (245, 16, 117),  # Pink
#         (16, 245, 117),  # Teal
#         (117, 16, 245),  # Purple
#         (245, 245, 16),   # Yellow
#         (128, 0, 128),   # Purple
#         (255, 192, 203), # Light Pink
#         (0, 255, 255),   # Cyan
#         (255, 165, 0),   # Orange4
#         (128, 128, 128),  # Gray
#         (245, 117, 16),  # Orange
#         (117, 245, 16),  # Lime Green
#     ]

#     output_frame = input_frame.copy()

#     num_actions = len(actions)

#     space_height = 4
#     total_space_height = (num_actions + 1) * space_height

#     bar_height = (frame_height - total_space_height) // num_actions

#     font_scale = max(0.4, bar_height / 25)
#     font_thickness = max(1, int(font_scale * 1.5))

#     for num, prob in enumerate(res):
#         bar_top = space_height + num * (bar_height + space_height)
#         bar_bottom = bar_top + bar_height

#         # Create an overlay for the semi-transparent rectangle
#         overlay = output_frame.copy()
#         cv.rectangle(overlay, (0, bar_top), (int(prob * frame_width), bar_bottom), colors[num], -1)

#         # Blend the overlay with the original frame
#         cv.addWeighted(overlay, opacity, output_frame, 1 - opacity, 0, output_frame)

#         # Draw the text
#         cv.putText(output_frame, actions[num], (10, bar_bottom - space_height // 2), cv.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv.LINE_AA)

#     return output_frame

# async def speak_async(words, on_done=None):
#     async with asyncio.Lock():
#         tts = gTTS(text=words, lang='id')
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmpfile:
#             tts.save(tmpfile.name)
#             filename = tmpfile.name

#         def play_audio(filename):

#             data, fs = sf.read(filename, dtype='float32')
#             sd.play(data, fs)
#             sd.wait()  # Wait until file is played
#             os.unlink(filename)  # Delete the temp file after playback
#             if on_done:
#                 on_done()

#         # Run the blocking play_audio function in a separate thread
#         loop = asyncio.get_running_loop()
#         await loop.run_in_executor(None, play_audio, filename)
#         print("Audio has been played.")

# def extract_keypoints_normalize(results):
#     midpoint_shoulder_x, midpoint_shoulder_y = 0, 0
#     shoulder_length = 1

#     if results.pose_landmarks:
#         left_shoulder = results.pose_landmarks.landmark[11]
#         right_shoulder = results.pose_landmarks.landmark[12]

#         midpoint_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
#         midpoint_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

#         shoulder_length = math.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)

#         selected_pose_landmarks = results.pose_landmarks.landmark[11:23]
#         pose = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length, 
#                           (res.y - midpoint_shoulder_y) / shoulder_length] for res in selected_pose_landmarks]).flatten()
#     else:
#         pose = np.zeros(12 * 2)

#     if results.left_hand_landmarks:
#         left_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length, 
#                                (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.left_hand_landmarks.landmark]).flatten()
#     else:
#         left_hand = np.zeros(21 * 2)

#     if results.right_hand_landmarks:
#         right_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length, 
#                                 (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.right_hand_landmarks.landmark]).flatten()
#     else:
#         right_hand = np.zeros(21 * 2)

#     return np.concatenate([pose, left_hand, right_hand])







# CLASS FOR MEDIAPIPE
class MediaPipeTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)  
        self.prev_frame_time = time.time()      
        # self.new_frame_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        global shared_fps

        new_frame_time = time.time()

        img = frame.to_ndarray(format="bgr24")  # Convert the frame to an ndarray

        # Process the image and get the pose
        img, results = media_pipe_detection(img, self.holistic)
        
        # Draw the landmarks
        draw_land_marks(img, results)
    
        # Convert the processed image back to a VideoFrame
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")

        fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time

        with lock:
            shared_fps = fps

        
        # if 'fps' in st.session_state:
        #     st.write(st.session_state['fps'])
        # else:
        #     st.write("FPS: Calculating...")

        return new_frame

# def media_pipe_detection(image, model):
#     image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
#     image.flags.writeable = False
#     results = model.process(image)
#     image.flags.writeable = True
#     image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
#     return image, results   

# def draw_land_marks(image, results):
#    custom_pose_connections = list(mp_pose.POSE_CONNECTIONS)

#    excluded_landmarks = [
#         PoseLandmark.NOSE,
#         PoseLandmark.LEFT_EYE_INNER,
#         PoseLandmark.LEFT_EYE,
#         PoseLandmark.LEFT_EYE_OUTER,
#         PoseLandmark.RIGHT_EYE_INNER,
#         PoseLandmark.RIGHT_EYE,
#         PoseLandmark.RIGHT_EYE_OUTER,
#         PoseLandmark.LEFT_EAR,
#         PoseLandmark.RIGHT_EAR,
#         PoseLandmark.MOUTH_LEFT,
#         PoseLandmark.MOUTH_RIGHT,
#         PoseLandmark.LEFT_HIP,
#         PoseLandmark.RIGHT_HIP,
#         PoseLandmark.LEFT_KNEE,
#         PoseLandmark.RIGHT_KNEE,
#         PoseLandmark.LEFT_ANKLE,
#         PoseLandmark.RIGHT_ANKLE,
#         PoseLandmark.LEFT_HEEL,
#         PoseLandmark.RIGHT_HEEL,
#         PoseLandmark.LEFT_FOOT_INDEX,
#         PoseLandmark.RIGHT_FOOT_INDEX
#     ]

#    for landmark in excluded_landmarks:
#         custom_pose_connections = [
#             connection_tuple for connection_tuple in custom_pose_connections if landmark.value not in connection_tuple]

#    mp_drawing.draw_landmarks(
#         image, results.pose_landmarks, connections=custom_pose_connections)
#    mp_drawing.draw_landmarks(
#         image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
#    mp_drawing.draw_landmarks(
#         image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)