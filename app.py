import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.framework.formats import landmark_pb2
import av
import threading
import queue
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2

import utils.settings as settings
import utils.helper as helper

lock = threading.Lock()
result = queue.Queue()

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

lstm_model = Sequential()

lstm_model.add(TimeDistributed(Dense(units=128, activation='tanh'), input_shape=(30, 108)))
lstm_model.add(LSTM(128, return_sequences=True, activation='tanh'))
lstm_model.add(Dropout(0.5))
lstm_model.add(LSTM(64, return_sequences=False, activation='tanh'))
lstm_model.add(Dropout(0.5))
lstm_model.add(Dense(32, activation='relu'))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(settings.MODEL_ACTIONS.shape[0], activation='softmax'))

lstm_model.load_weights(settings.LSTM_MODEL)

sequence = []
sentence = []
predictions = []

prev_frame_time = time.time()

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
    global result

    new_frame_time = time.time()

    img = frame.to_ndarray(format="bgr24")

    img, results = media_pipe_detection(img, holistic_model)

    draw_land_marks(img, results)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # Log FPS or update an external metric
    print(fps)
    result.put(fps)

    # with lock:
    #     result = fps

    # Convert the processed image back to a VideoFrame
    new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")

    return new_frame

def extract_keypoints_normalize(results):
    midpoint_shoulder_x, midpoint_shoulder_y = 0, 0
    shoulder_length = 1

    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[11]
        right_shoulder = results.pose_landmarks.landmark[12]

        midpoint_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
        midpoint_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

        shoulder_length = math.sqrt((left_shoulder.x - right_shoulder.x) ** 2 + (left_shoulder.y - right_shoulder.y) ** 2)

        selected_pose_landmarks = results.pose_landmarks.landmark[11:23]
        pose = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length, 
                          (res.y - midpoint_shoulder_y) / shoulder_length] for res in selected_pose_landmarks]).flatten()
    else:
        pose = np.zeros(12 * 2)

    if results.left_hand_landmarks:
        left_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length, 
                               (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 2)

    if results.right_hand_landmarks:
        right_hand = np.array([[(res.x - midpoint_shoulder_x) / shoulder_length, 
                                (res.y - midpoint_shoulder_y) / shoulder_length] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 2)

    return np.concatenate([pose, left_hand, right_hand])

def load_lstm_model():
    model = Sequential()

    model.add(TimeDistributed(Dense(units=128, activation='tanh'), input_shape=(30, 108)))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(settings.MODEL_ACTIONS.shape[0], activation='softmax'))

    model.load_weights(settings.LSTM_MODEL)

    return model 

def lstm_callback(frame):
    global sequence
    global sentence
    global predictions

    threshold = 0.5 

    img = frame.to_ndarray(format="bgr24")

    img, results = media_pipe_detection(img, holistic_model)

    draw_land_marks(img, results)
   
    actions = np.array(["maaf", "tolong", "nama", "saya", "siapa", "rumah", "start", "standby", "delete"])

    keypoints = extract_keypoints_normalize(results)

    sequence.append(keypoints)

    sequence = sequence[-30:]

    print(len(sequence))


    if len(sequence) == 30:
        sequence_array = np.array(sequence)
        if sequence_array.shape == (30, 108):
            res = lstm_model.predict(np.expand_dims(sequence_array, axis=0))[0]
            print(actions[np.argmax(res)])
            print(res)
            print("")
            predictions.append(np.argmax(res))
                
            if np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                            result.put(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])
                        result.put(actions[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]
                

    new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")

    return new_frame


# STREAMLIT UI

# TITLE AND HEADER
st.set_page_config(
    page_title="BISINDO Translator",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# SIDEBAR
st.sidebar.title("Menus")
st.sidebar.markdown("""
    <style>
    .wrapper {
        word-wrap: break-word;
    }
    </style>
    <div class="wrapper">
        Please select translator mode
    </div>
""", unsafe_allow_html=True)
mode_type = st.sidebar.selectbox(
    "", [settings.VIDEO, settings.WEBCAM, settings.WEBCAM_MEDIAPIPE])

# HANDLE MENUS
col1, col2, col3= st.columns([0.1,0.8, 0.1])

col1.empty()
with col2.container():
    st.title("BISINDO Translator Using Long Short Term Memory (LSTM)")

    if mode_type == settings.VIDEO:
        st.markdown("""
        <style>
        .wrapper {
            word-wrap: break-word;
        }
        </style>
        <div class="wrapper">
            Under maintenance.&nbsp
            For translator, we recommend you to use the webcam first for better experience
        </div>
    """, unsafe_allow_html=True)
        
    elif mode_type == settings.WEBCAM:
        ctx = webrtc_streamer(key="example", 
                        rtc_configuration=settings.RTC_CONFIGURATION, 
                        video_frame_callback=mediapipe_callback,
                    media_stream_constraints={"video": {"width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}, 
                                            "audio": False})
        
        if ctx.state.playing:
            result_placeholder = st.empty()

            while True:
                fps = result.get()
                result_placeholder.markdown(fps)

        #  webrtc_streamer(key="example", 
        #                 rtc_configuration=settings.RTC_CONFIGURATION, 
        #                video_transformer_factory=helper.MediaPipeTransformer,
        #              media_stream_constraints={"video": {"width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}, 
        #                                     "audio": False})
        
    elif mode_type == settings.WEBCAM_MEDIAPIPE:
        ctx = webrtc_streamer(key="example", 
                        rtc_configuration=settings.RTC_CONFIGURATION, 
                        video_frame_callback=lstm_callback,
                    media_stream_constraints={"video": {"width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}, 
                                            "audio": False})
        
        if ctx.state.playing:
            result_placeholder = st.empty()

            while True:
                word = result.get()
                result_placeholder.markdown(word)

        # webrtc_streamer(key="example", 
        #                 rtc_configuration=settings.RTC_CONFIGURATION, 
        #                video_transformer_factory=helper.MediaPipeTransformer,
        #              media_stream_constraints={"video": {"width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}, 
        #                                     "audio": False})
    



col3.empty()

# STREAMLIT UI


