"""
Camera Manager Module
Coordinates multiple camera streams and processing
Connects cameras to detection and recognition pipeline
"""

import time
import threading
from typing import Dict, List
import logging
import numpy as np

import config
from streams.rtsp_handler import MultiStreamManager
from detection.face_detector import FaceDetector
from recognition.embedding import FaceEmbedder
from recognition.matcher import FaceMatcher
from fusion.fusion_engine import FusionEngine
from database.db import get_db


class CameraProcessor:
    """
    Processes frames from a single camera
    Runs detection and recognition pipeline
    """
    
    def __init__(self, camera_id: str, location: str, fusion_engine: FusionEngine):
        self.camera_id = camera_id
        self.location = location
        self.fusion_engine = fusion_engine
        
        # Pipeline components
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        self.matcher = FaceMatcher()
        
        # Processing state
        self.is_running = False
        self.process_thread = None
        
        # Statistics
        self.frames_processed = 0
        self.faces_detected = 0
        self.faces_recognized = 0
        
        # Logging
        self.logger = logging.getLogger(f'CameraProcessor-{camera_id}')
        self.logger.setLevel(logging.INFO)
        
        print(f"[{self.camera_id}] Processor initialized")
    
    def start(self, stream_handler):
        """
        Start processing frames from stream
        
        Args:
            stream_handler: StreamHandler instance
        """
        if self.is_running:
            print(f"[{self.camera_id}] Already processing")
            return
        
        self.is_running = True
        self.stream_handler = stream_handler
        self.process_thread = threading.Thread(
            target=self._process_loop,
            daemon=True
        )
        self.process_thread.start()
        
        print(f"[{self.camera_id}] Started processing")
    
    def stop(self):
        """Stop processing"""
        print(f"[{self.camera_id}] Stopping processor...")
        
        self.is_running = False
        
        if self.process_thread:
            self.process_thread.join(timeout=5)
        
        print(f"[{self.camera_id}] Processor stopped")
    
    def _process_loop(self):
        """Main processing loop"""
        print(f"[{self.camera_id}] Processing loop started")
        
        while self.is_running:
            try:
                # Get frame from stream
                frame = self.stream_handler.get_frame(timeout=1.0)
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Process frame
                self._process_frame(frame)
                
            except Exception as e:
                self.logger.error(f"Processing error: {e}")
                time.sleep(0.5)
        
        print(f"[{self.camera_id}] Processing loop stopped")
    
    def _process_frame(self, frame: np.ndarray):
        """
        Process a single frame
        
        Args:
            frame: Video frame (BGR)
        """
        self.frames_processed += 1
        
        # Detect faces
        detections = self.detector.detect_faces(frame)
        
        if not detections:
            return
        
        self.faces_detected += len(detections)
        
        # Process each detected face
        for detection in detections:
            self._process_face(frame, detection)
    
    def _process_face(self, frame: np.ndarray, detection: Dict):
        """
        Process a detected face
        
        Args:
            frame: Video frame
            detection: Face detection data
        """
        # Extract face ROI
        face_roi = self.detector.extract_face_roi(frame, detection['box'])
        
        if face_roi is None:
            return
        
        # Generate embedding
        embedding = self.embedder.generate_embedding(face_roi)
        
        if embedding is None:
            return
        
        # Match face
        recognition_result = self.matcher.recognize_face(embedding)
        
        if recognition_result is None:
            return
        
        # Get face location for logging
        x, y, w, h = detection['box']
        face_location = {'x': x, 'y': y, 'w': w, 'h': h}
        
        if recognition_result['recognized']:
            # Recognized identity
            identity = recognition_result['identity']
            register_number = identity['register_number']
            similarity = recognition_result['similarity']
            
            self.faces_recognized += 1
            
            # Send to fusion engine
            self.fusion_engine.add_detection(
                camera_id=self.camera_id,
                register_number=register_number,
                confidence=similarity,
                location=self.location,
                face_location=face_location
            )
            
            self.logger.debug(
                f"Recognized: {register_number} | "
                f"Confidence: {similarity:.3f}"
            )
        else:
            # Unknown face
            self.fusion_engine.add_unknown_detection(
                camera_id=self.camera_id,
                confidence=detection['confidence'],
                face_location=face_location
            )
            
            self.logger.debug("Unknown face detected")
    
    def get_statistics(self) -> Dict:
        """Get processor statistics"""
        return {
            'camera_id': self.camera_id,
            'location': self.location,
            'is_running': self.is_running,
            'frames_processed': self.frames_processed,
            'faces_detected': self.faces_detected,
            'faces_recognized': self.faces_recognized,
            'recognition_rate': (
                self.faces_recognized / self.faces_detected 
                if self.faces_detected > 0 else 0
            )
        }


class CameraManager:
    """
    Central manager for all cameras
    Coordinates streams, processing, and fusion
    """
    
    def __init__(self):
        self.db = get_db()
        
        # Stream management
        self.stream_manager = MultiStreamManager()
        
        # Fusion engine
        self.fusion_engine = FusionEngine()
        
        # Camera processors
        self.processors = {}
        
        # Statistics
        self.start_time = None
        
        # Logging
        self._setup_logging()
        
        print("[CameraManager] Initialized")
    
    def _setup_logging(self):
        """Setup system logging"""
        self.logger = logging.getLogger('CameraManager')
        self.logger.setLevel(logging.INFO)
        
        if config.LOG_TO_FILE:
            handler = logging.FileHandler(config.SYSTEM_LOG)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def initialize_cameras(self):
        """Initialize all configured cameras"""
        print("\n[CameraManager] Initializing cameras...")
        print("=" * 60)
        
        for cam_config in config.CAMERA_CONFIGS:
            if not cam_config.get('enabled', True):
                continue
            
            camera_id = cam_config['camera_id']
            
            # Add to stream manager
            self.stream_manager.add_camera(cam_config)
            
            # Create processor
            processor = CameraProcessor(
                camera_id,
                cam_config['location'],
                self.fusion_engine
            )
            self.processors[camera_id] = processor
            
            # Update database
            self.db.update_camera_status(
                camera_id,
                'initialized',
                cam_config['location']
            )
            
            print(f"✓ Initialized: {camera_id} ({cam_config['location']})")
        
        print("=" * 60)
        print(f"Total cameras initialized: {len(self.processors)}")
    
    def start_all(self):
        """Start all cameras and processing"""
        print("\n[CameraManager] Starting system...")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Start fusion engine
        self.fusion_engine.start()
        print("✓ Fusion engine started")
        
        # Start all streams
        self.stream_manager.start_all()
        time.sleep(2)  # Give streams time to connect
        print("✓ Camera streams started")
        
        # Start processors
        for camera_id, processor in self.processors.items():
            stream = self.stream_manager.get_stream(camera_id)
            if stream and stream.is_alive():
                processor.start(stream)
                self.db.update_camera_status(camera_id, 'online')
                print(f"✓ Processing started: {camera_id}")
            else:
                print(f"✗ Stream not available: {camera_id}")
                self.db.update_camera_status(camera_id, 'offline')
        
        print("=" * 60)
        print("✓ System started successfully")
        print("\nMonitoring attendance... (Press Ctrl+C to stop)")
    
    def stop_all(self):
        """Stop all cameras and processing"""
        print("\n[CameraManager] Stopping system...")
        print("=" * 60)
        
        # Stop processors
        for camera_id, processor in self.processors.items():
            processor.stop()
            self.db.update_camera_status(camera_id, 'offline')
        
        print("✓ Processors stopped")
        
        # Stop streams
        self.stream_manager.stop_all()
        print("✓ Streams stopped")
        
        # Stop fusion engine
        self.fusion_engine.stop()
        print("✓ Fusion engine stopped")
        
        print("=" * 60)
        print("✓ System stopped")
    
    def print_statistics(self):
        """Print system statistics"""
        print("\n" + "=" * 60)
        print("SYSTEM STATISTICS")
        print("=" * 60)
        
        # Uptime
        if self.start_time:
            uptime = time.time() - self.start_time
            hours = int(uptime // 3600)
            minutes = int((uptime % 3600) // 60)
            seconds = int(uptime % 60)
            print(f"Uptime: {hours}h {minutes}m {seconds}s")
        
        print()
        
        # Stream statistics
        stream_stats = self.stream_manager.get_system_stats()
        print(f"Cameras: {stream_stats['active_cameras']}/{stream_stats['total_cameras']} active")
        print()
        
        # Processor statistics
        print("Camera Processing:")
        for camera_id, processor in self.processors.items():
            stats = processor.get_statistics()
            print(f"  {camera_id}:")
            print(f"    Frames: {stats['frames_processed']}")
            print(f"    Faces Detected: {stats['faces_detected']}")
            print(f"    Faces Recognized: {stats['faces_recognized']}")
            print(f"    Recognition Rate: {stats['recognition_rate']:.2%}")
        
        print()
        
        # Fusion statistics
        fusion_stats = self.fusion_engine.get_statistics()
        print("Fusion Engine:")
        print(f"  Total Detections: {fusion_stats['attendance_engine']['total_detections']}")
        print(f"  Total Recognitions: {fusion_stats['attendance_engine']['total_recognitions']}")
        print(f"  Attendance Marked: {fusion_stats['attendance_engine']['total_attendance_marked']}")
        print(f"  Duplicates Prevented: {fusion_stats['attendance_engine']['total_duplicates_prevented']}")
        
        print()
        
        # Today's attendance
        summary = self.fusion_engine.get_today_summary()
        print("Today's Attendance:")
        print(f"  Date: {summary['date']}")
        print(f"  Total Present: {summary['total_present']}")
        
        print("=" * 60)
    
    def print_today_attendance(self):
        """Print today's attendance list"""
        summary = self.fusion_engine.get_today_summary()
        
        print("\n" + "=" * 60)
        print("TODAY'S ATTENDANCE")
        print("=" * 60)
        print(f"Date: {summary['date']}")
        print(f"Total Present: {summary['total_present']}")
        print()
        
        if summary['total_present'] > 0:
            print(f"{'No.':<5} {'Register':<15} {'Name':<25} {'Time':<10} {'Conf':<6}")
            print("-" * 60)
            
            for i, record in enumerate(summary['records'], 1):
                check_in = record['check_in_time']
                if isinstance(check_in, str):
                    time_str = check_in.split()[1][:8]  # Extract time
                else:
                    time_str = check_in.strftime('%H:%M:%S')
                
                print(f"{i:<5} {record['register_number']:<15} {record['name']:<25} "
                      f"{time_str:<10} {record['confidence_score']:.2f}")
        else:
            print("No attendance marked yet today")
        
        print("=" * 60)
    
    def health_check(self):
        """Perform system health check"""
        print("\n[CameraManager] Performing health check...")
        
        # Check streams
        self.stream_manager.health_check()
        
        # Check processors
        print("\nProcessor Status:")
        print("=" * 60)
        for camera_id, processor in self.processors.items():
            status = "🟢 RUNNING" if processor.is_running else "🔴 STOPPED"
            print(f"  {camera_id}: {status}")
        
        print("=" * 60)
