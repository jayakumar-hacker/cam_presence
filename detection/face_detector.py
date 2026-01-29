"""
Face Detection Module
Multi-face detection using MTCNN or OpenCV
Returns face bounding boxes and landmarks
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import time

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("[WARNING] MTCNN not available, falling back to OpenCV")

import config


class FaceDetector:
    """
    Face detection engine supporting multiple backends
    Primary: MTCNN for accurate multi-face detection
    Fallback: OpenCV Haar Cascade
    """
    
    def __init__(self, model: str = config.DETECTION_MODEL):
        self.model = model
        self.detector = None
        
        self._initialize_detector()
        
        # Statistics
        self.total_detections = 0
        self.total_frames = 0
        
        print(f"[FaceDetector] Initialized with model: {self.model}")
    
    def _initialize_detector(self):
        """Initialize the detection model"""
        if self.model == "mtcnn" and MTCNN_AVAILABLE:
            try:
                self.detector = MTCNN(
                    min_face_size=config.MIN_FACE_SIZE,
                    steps_threshold=[0.6, 0.7, 0.8],
                    scale_factor=0.709
                )
                print("[FaceDetector] MTCNN loaded successfully")
            except Exception as e:
                print(f"[FaceDetector] MTCNN initialization failed: {e}")
                self.model = "opencv"
        
        if self.model == "opencv" or self.detector is None:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
            
            if self.detector.empty():
                raise RuntimeError("Failed to load OpenCV face detector")
            
            print("[FaceDetector] OpenCV Haar Cascade loaded successfully")
    
    def detect_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Detect faces in frame
        
        Args:
            frame: Input image/frame (BGR format)
        
        Returns:
            List of face detections, each containing:
            - box: (x, y, width, height)
            - confidence: detection confidence score
            - keypoints: facial landmarks (if available)
        """
        if frame is None or frame.size == 0:
            return []
        
        self.total_frames += 1
        
        if self.model == "mtcnn":
            return self._detect_mtcnn(frame)
        else:
            return self._detect_opencv(frame)
    
    def _detect_mtcnn(self, frame: np.ndarray) -> List[dict]:
        """Detect faces using MTCNN"""
        try:
            # Convert BGR to RGB for MTCNN
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = self.detector.detect_faces(rgb_frame)
            
            if not detections:
                return []
            
            faces = []
            for detection in detections:
                confidence = detection['confidence']
                
                # Filter by confidence threshold
                if confidence < config.DETECTION_CONFIDENCE:
                    continue
                
                # Extract bounding box
                box = detection['box']
                x, y, w, h = box
                
                # Ensure positive dimensions
                if w <= 0 or h <= 0:
                    continue
                
                # Ensure face is large enough
                if w < config.MIN_FACE_SIZE or h < config.MIN_FACE_SIZE:
                    continue
                
                # Extract keypoints
                keypoints = detection.get('keypoints', {})
                
                face_data = {
                    'box': (x, y, w, h),
                    'confidence': confidence,
                    'keypoints': keypoints,
                    'detector': 'mtcnn'
                }
                
                faces.append(face_data)
            
            self.total_detections += len(faces)
            return faces
            
        except Exception as e:
            print(f"[FaceDetector] MTCNN detection error: {e}")
            return []
    
    def _detect_opencv(self, frame: np.ndarray) -> List[dict]:
        """Detect faces using OpenCV Haar Cascade"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces_rects = self.detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(config.MIN_FACE_SIZE, config.MIN_FACE_SIZE),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces = []
            for (x, y, w, h) in faces_rects:
                face_data = {
                    'box': (x, y, w, h),
                    'confidence': 0.95,  # OpenCV doesn't provide confidence
                    'keypoints': {},
                    'detector': 'opencv'
                }
                faces.append(face_data)
            
            self.total_detections += len(faces)
            return faces
            
        except Exception as e:
            print(f"[FaceDetector] OpenCV detection error: {e}")
            return []
    
    def extract_face_roi(self, frame: np.ndarray, box: Tuple[int, int, int, int],
                        padding: float = 0.2) -> Optional[np.ndarray]:
        """
        Extract face region of interest with padding
        
        Args:
            frame: Input frame
            box: (x, y, width, height)
            padding: Padding ratio (0.2 = 20% padding)
        
        Returns:
            Face ROI image or None
        """
        if frame is None or frame.size == 0:
            return None
        
        x, y, w, h = box
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(frame.shape[1], x + w + pad_w)
        y2 = min(frame.shape[0], y + h + pad_h)
        
        # Extract ROI
        face_roi = frame[y1:y2, x1:x2]
        
        if face_roi.size == 0:
            return None
        
        return face_roi
    
    def draw_detections(self, frame: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        Draw detection boxes on frame (for debugging)
        
        Args:
            frame: Input frame
            detections: List of face detections
        
        Returns:
            Frame with drawn boxes
        """
        output_frame = frame.copy()
        
        for detection in detections:
            x, y, w, h = detection['box']
            confidence = detection['confidence']
            
            # Draw rectangle
            color = (0, 255, 0)  # Green
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw confidence
            label = f"{confidence:.2f}"
            cv2.putText(output_frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw keypoints if available
            keypoints = detection.get('keypoints', {})
            for name, point in keypoints.items():
                if point:
                    cv2.circle(output_frame, tuple(point), 2, (0, 0, 255), -1)
        
        return output_frame
    
    def get_stats(self) -> dict:
        """Get detector statistics"""
        return {
            'model': self.model,
            'total_frames': self.total_frames,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': (
                self.total_detections / self.total_frames 
                if self.total_frames > 0 else 0
            )
        }


class BatchFaceDetector:
    """
    Batch processor for detecting faces across multiple frames
    Used for efficient processing of frame streams
    """
    
    def __init__(self):
        self.detector = FaceDetector()
        print("[BatchFaceDetector] Initialized")
    
    def process_batch(self, frames: List[np.ndarray]) -> List[List[dict]]:
        """
        Process batch of frames
        
        Args:
            frames: List of frames to process
        
        Returns:
            List of detection results for each frame
        """
        results = []
        
        for frame in frames:
            detections = self.detector.detect_faces(frame)
            results.append(detections)
        
        return results
    
    def process_stream(self, frames_iterator, max_faces: int = config.MAX_FACES_PER_FRAME):
        """
        Process stream of frames (generator)
        
        Args:
            frames_iterator: Iterator yielding frames
            max_faces: Maximum faces to detect per frame
        
        Yields:
            (frame, detections) tuples
        """
        for frame in frames_iterator:
            detections = self.detector.detect_faces(frame)
            
            # Limit number of faces
            if len(detections) > max_faces:
                # Sort by confidence and keep top N
                detections = sorted(
                    detections, 
                    key=lambda x: x['confidence'], 
                    reverse=True
                )[:max_faces]
            
            yield frame, detections
