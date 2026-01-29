"""
Face detection module
Multi-face detection using MTCNN and OpenCV
"""

from .face_detector import FaceDetector, BatchFaceDetector

__all__ = ['FaceDetector', 'BatchFaceDetector']
