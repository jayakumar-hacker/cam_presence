"""
Configuration file for Smart CCTV Attendance System
Central configuration for all system parameters
"""

import os
from pathlib import Path

# ===========================
# PATH CONFIGURATION
# ===========================
BASE_DIR = Path(__file__).parent
DATABASE_PATH = BASE_DIR / "database" / "attendance.db"
LOGS_DIR = BASE_DIR / "logs"
REGISTERED_FACES_DIR = BASE_DIR / "registered_faces"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
REGISTERED_FACES_DIR.mkdir(exist_ok=True)

# ===========================
# CAMERA CONFIGURATION
# ===========================
CAMERA_CONFIGS = [
    {
        "camera_id": "CAM_01",
        "stream_url": "http://192.168.1.100:8080/video",  # Example: IP Webcam
        "location": "Main Entrance",
        "enabled": True
    },
    {
        "camera_id": "CAM_02",
        "stream_url": "http://192.168.1.101:8080/video",  # Example: DroidCam
        "location": "Side Entrance",
        "enabled": True
    }
]

# Stream settings
STREAM_RECONNECT_DELAY = 5  # seconds
STREAM_READ_TIMEOUT = 10  # seconds
FRAME_SKIP = 2  # Process every Nth frame for performance

# ===========================
# DETECTION CONFIGURATION
# ===========================
DETECTION_MODEL = "mtcnn"  # Options: "mtcnn", "opencv", "dlib"
DETECTION_CONFIDENCE = 0.90  # Minimum confidence for face detection
MIN_FACE_SIZE = 40  # Minimum face size in pixels
MAX_FACES_PER_FRAME = 10  # Maximum faces to detect per frame

# ===========================
# RECOGNITION CONFIGURATION
# ===========================
RECOGNITION_MODEL = "facenet"  # Options: "facenet", "vggface", "arcface"
EMBEDDING_SIZE = 512  # Embedding vector size
RECOGNITION_THRESHOLD = 0.6  # Cosine similarity threshold (lower = stricter)
UNKNOWN_THRESHOLD = 0.7  # Above this = definitely unknown

# ===========================
# FUSION ENGINE CONFIGURATION
# ===========================
FUSION_TIME_WINDOW = 3.0  # seconds - aggregate detections within this window
FUSION_MIN_DETECTIONS = 3  # Minimum detections before making decision
FUSION_CONFIDENCE_THRESHOLD = 0.75  # Average confidence threshold
CROSS_CAMERA_VALIDATION = True  # Require multiple cameras to see same person

# ===========================
# ATTENDANCE CONFIGURATION
# ===========================
ATTENDANCE_COOLDOWN = 300  # seconds (5 minutes) - prevent duplicate marking
ATTENDANCE_SESSION_TIMEOUT = 28800  # seconds (8 hours) - max session duration
ATTENDANCE_MIN_CONFIDENCE = 0.8  # Minimum confidence to mark attendance
ATTENDANCE_REQUIRES_MULTIPLE_FRAMES = True  # Require multiple frame validation
ATTENDANCE_MIN_VALIDATION_FRAMES = 5  # Minimum frames for validation

# Attendance time windows
ATTENDANCE_START_TIME = "08:00"  # Daily attendance start
ATTENDANCE_END_TIME = "18:00"  # Daily attendance end

# ===========================
# DEDUPLICATION CONFIGURATION
# ===========================
DEDUP_STRATEGY = "identity_based"  # Options: "identity_based", "time_based"
DEDUP_TIME_WINDOW = 10  # seconds - merge detections within this window
DEDUP_SPATIAL_THRESHOLD = 100  # pixels - merge nearby detections

# ===========================
# CONFIDENCE TRACKING
# ===========================
CONFIDENCE_WINDOW_SIZE = 10  # Number of recent detections to track
CONFIDENCE_DECAY_RATE = 0.1  # Confidence decay per second
CONFIDENCE_MIN_SAMPLES = 3  # Minimum samples before making decision

# ===========================
# DATABASE CONFIGURATION
# ===========================
DB_POOL_SIZE = 5
DB_TIMEOUT = 30  # seconds

# ===========================
# LOGGING CONFIGURATION
# ===========================
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_TO_FILE = True
LOG_TO_CONSOLE = True
LOG_MAX_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 5

# Log file paths
SYSTEM_LOG = LOGS_DIR / "system.log"
DETECTION_LOG = LOGS_DIR / "detection.log"
RECOGNITION_LOG = LOGS_DIR / "recognition.log"
ATTENDANCE_LOG = LOGS_DIR / "attendance.log"
FUSION_LOG = LOGS_DIR / "fusion.log"

# ===========================
# PERFORMANCE CONFIGURATION
# ===========================
MAX_WORKER_THREADS = 4  # Maximum threads for parallel processing
FRAME_BUFFER_SIZE = 30  # Maximum frames to buffer per camera
QUEUE_TIMEOUT = 1  # seconds

# ===========================
# SYSTEM CONFIGURATION
# ===========================
SYSTEM_NAME = "Smart CCTV Attendance System"
VERSION = "1.0.0"
DEBUG_MODE = False

# Display settings
SHOW_DETECTION_BOXES = False  # Set to True for debugging (shows frames)
SAVE_DETECTION_IMAGES = False  # Save detected faces for debugging

# ===========================
# REGISTRATION CONFIGURATION
# ===========================
REGISTRATION_SAMPLES_REQUIRED = 5  # Number of face samples needed for registration
REGISTRATION_QUALITY_THRESHOLD = 0.85  # Minimum quality for registration sample
REGISTRATION_TIMEOUT = 60  # seconds - max time for registration session
