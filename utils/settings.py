from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
# IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
WEBCAM_MEDIAPIPE = 'Webcam (Mediapipe)'
# RTMP = 'RTMP'
# YOUTUBE = 'YouTube'

# SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTMP, YOUTUBE]
SOURCES_LIST = [VIDEO, WEBCAM, WEBCAM_MEDIAPIPE]

# Images config
# IMAGES_DIR = ROOT / 'img'
# DEFAULT_IMAGE = IMAGES_DIR / 'bus.jpg'
# DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'office_4_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
# VIDEO_1_PATH = VIDEO_DIR / '20km.MOV'
# VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
# VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
# VIDEOS_DICT = {
#     'video_1': VIDEO_1_PATH,
#     'video_2': VIDEO_2_PATH,
#     'video_3': VIDEO_3_PATH,
# }

# ML Model config
MODEL_DIR = ROOT / 'models'
LSTM_MODEL = MODEL_DIR / 'model3.h5'

# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

# Video Settings
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480