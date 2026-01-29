"""
Camera Management Module
======================

Coordinates multiple camera streams and processing pipelines.

Components:
    - CameraManager: Central coordinator for all cameras
    - CameraProcessor: Per-camera frame processing

Usage:
    from cams import CameraManager
    
    manager = CameraManager()
    manager.initialize_cameras()
    manager.start_all()
"""

from .cam_manager import CameraManager, CameraProcessor

__version__ = '1.0.0'
__author__ = 'Smart CCTV Team'
__all__ = ['CameraManager', 'CameraProcessor']

# Module-level configuration
DEFAULT_CAMERA_COUNT = 2
