import streamlit as st
import utils.settings as settings
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import math
from mediapipe.framework.formats import landmark_pb2
import av

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

class MediaPipeTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)        
        # self.new_frame_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")  # Convert the frame to an ndarray

        # Process the image and get the pose
        img, results = media_pipe_detection(img, self.holistic)
        
        # Draw the landmarks
        draw_land_marks(img, results)

        # Convert the processed image back to a VideoFrame
        new_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        return new_frame

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

# TITLE AND HEADER
st.set_page_config(
    page_title="BISINDO Translator",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("BISINDO Translator Using Long Short Term Memory")

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
    webrtc_streamer(key="example", 
                    rtc_configuration=RTC_CONFIGURATION, 
                    video_processor_factory=MediaPipeTransformer, 
                media_stream_constraints={"video": {"width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}, 
                                          "audio": False})

elif mode_type == settings.WEBCAM_MEDIAPIPE:
     webrtc_streamer(key="example", 
                    rtc_configuration=RTC_CONFIGURATION, 
                    video_processor_factory=MediaPipeTransformer, 
                media_stream_constraints={"video": {"width": settings.VIDEO_WIDTH, "height": settings.VIDEO_HEIGHT}, 
                                          "audio": False})
    
