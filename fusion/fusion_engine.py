"""
Fusion Engine Module
Central multi-camera fusion for attendance decisions
Aggregates detections from all cameras
"""

import time
import threading
from queue import Queue, Empty
from typing import Dict, List, Optional
from collections import defaultdict
import logging

import config
from database.db import get_db
from attendance.attendance_engine import AttendanceEngine


class FusionEngine:
    """
    Central fusion engine for multi-camera system
    Aggregates detections from all cameras
    Makes unified attendance decisions
    """
    
    def __init__(self):
        self.db = get_db()
        self.attendance_engine = AttendanceEngine()
        
        # Detection queue from all cameras
        self.detection_queue = Queue(maxsize=1000)
        
        # Processing thread
        self.is_running = False
        self.fusion_thread = None
        
        # Detection aggregation
        # {register_number: [detection_events]}
        self.detection_buffer = defaultdict(list)
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.total_fused = 0
        self.total_decisions = 0
        
        # Setup logging
        self._setup_logging()
        
        print("[FusionEngine] Initialized")
        print(f"[FusionEngine] Fusion window: {config.FUSION_TIME_WINDOW}s")
        print(f"[FusionEngine] Min detections: {config.FUSION_MIN_DETECTIONS}")
    
    def _setup_logging(self):
        """Setup fusion logging"""
        self.logger = logging.getLogger('FusionEngine')
        self.logger.setLevel(logging.INFO)
        
        if config.LOG_TO_FILE:
            handler = logging.FileHandler(config.FUSION_LOG)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def start(self):
        """Start fusion engine"""
        if self.is_running:
            print("[FusionEngine] Already running")
            return
        
        self.is_running = True
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.fusion_thread.start()
        
        print("[FusionEngine] Started")
    
    def stop(self):
        """Stop fusion engine"""
        print("[FusionEngine] Stopping...")
        
        self.is_running = False
        
        if self.fusion_thread:
            self.fusion_thread.join(timeout=5)
        
        print("[FusionEngine] Stopped")
    
    def add_detection(self, camera_id: str, register_number: str,
                     confidence: float, location: str,
                     face_location: Dict = None):
        """
        Add detection from camera
        
        Args:
            camera_id: Camera identifier
            register_number: Student register number
            confidence: Recognition confidence
            location: Camera location
            face_location: Face bounding box
        """
        detection = {
            'camera_id': camera_id,
            'register_number': register_number,
            'confidence': confidence,
            'location': location,
            'face_location': face_location,
            'timestamp': time.time()
        }
        
        try:
            self.detection_queue.put_nowait(detection)
        except:
            # Queue full, drop detection
            self.logger.warning("Detection queue full, dropping detection")
    
    def add_unknown_detection(self, camera_id: str, confidence: float,
                             face_location: Dict = None):
        """
        Add unknown face detection
        
        Args:
            camera_id: Camera identifier
            confidence: Detection confidence
            face_location: Face bounding box
        """
        self.attendance_engine.process_unknown_detection(
            camera_id, confidence, face_location
        )
    
    def _fusion_loop(self):
        """Main fusion processing loop"""
        print("[FusionEngine] Fusion loop started")
        
        while self.is_running:
            try:
                # Get detection from queue
                detection = self.detection_queue.get(timeout=0.1)
                
                # Process detection
                self._process_detection(detection)
                
            except Empty:
                # No detection available, check for timeouts
                self._check_timeouts()
                time.sleep(0.1)
            
            except Exception as e:
                self.logger.error(f"Fusion loop error: {e}")
                time.sleep(0.1)
        
        print("[FusionEngine] Fusion loop stopped")
    
    def _process_detection(self, detection: Dict):
        """
        Process a single detection event
        
        Args:
            detection: Detection event data
        """
        register_number = detection['register_number']
        
        with self.buffer_lock:
            # Add to buffer
            self.detection_buffer[register_number].append(detection)
            
            # Check if ready for fusion
            if self._should_fuse(register_number):
                self._fuse_and_decide(register_number)
    
    def _should_fuse(self, register_number: str) -> bool:
        """
        Check if detections should be fused
        
        Args:
            register_number: Student register number
        
        Returns:
            True if ready to fuse
        """
        detections = self.detection_buffer.get(register_number, [])
        
        if len(detections) < config.FUSION_MIN_DETECTIONS:
            return False
        
        # Check time span
        timestamps = [d['timestamp'] for d in detections]
        time_span = max(timestamps) - min(timestamps)
        
        # Ready if we have enough detections within time window
        return time_span >= config.FUSION_TIME_WINDOW
    
    def _fuse_and_decide(self, register_number: str):
        """
        Fuse detections and make attendance decision
        
        Args:
            register_number: Student register number
        """
        detections = self.detection_buffer[register_number]
        
        if not detections:
            return
        
        # Extract data
        cameras = set(d['camera_id'] for d in detections)
        locations = set(d['location'] for d in detections)
        confidences = [d['confidence'] for d in detections]
        
        # Calculate statistics
        total_detections = len(detections)
        camera_count = len(cameras)
        avg_confidence = sum(confidences) / len(confidences)
        max_confidence = max(confidences)
        
        # Decision logic
        decision = 'ignore'
        
        # Check confidence threshold
        if avg_confidence >= config.FUSION_CONFIDENCE_THRESHOLD:
            # Check cross-camera validation if enabled
            if config.CROSS_CAMERA_VALIDATION:
                if camera_count >= 2:
                    decision = 'mark_attendance'
                else:
                    decision = 'uncertain'
            else:
                decision = 'mark_attendance'
        else:
            decision = 'ignore'
        
        # Log fusion event
        self.db.log_fusion_event(
            student_id=None,  # Will be filled by attendance engine
            register_number=register_number,
            total_detections=total_detections,
            camera_count=camera_count,
            avg_confidence=avg_confidence,
            max_confidence=max_confidence,
            cameras=list(cameras),
            decision=decision
        )
        
        self.total_fused += 1
        
        self.logger.info(
            f"Fused: {register_number} | "
            f"Detections: {total_detections} | "
            f"Cameras: {camera_count} | "
            f"Avg Confidence: {avg_confidence:.3f} | "
            f"Decision: {decision}"
        )
        
        # Execute decision
        if decision == 'mark_attendance':
            # Use most recent detection for camera/location
            latest_detection = max(detections, key=lambda d: d['timestamp'])
            
            result = self.attendance_engine.process_detection(
                camera_id=latest_detection['camera_id'],
                register_number=register_number,
                confidence=avg_confidence,
                location=latest_detection['location'],
                face_location=latest_detection.get('face_location')
            )
            
            self.total_decisions += 1
            
            self.logger.info(
                f"Decision: {result['action']} | "
                f"Reason: {result['reason']}"
            )
        
        # Clear buffer
        self.detection_buffer[register_number].clear()
    
    def _check_timeouts(self):
        """Check for timed-out detection buffers"""
        current_time = time.time()
        timeout = config.FUSION_TIME_WINDOW * 2
        
        with self.buffer_lock:
            identities_to_process = []
            
            for reg_num, detections in self.detection_buffer.items():
                if not detections:
                    continue
                
                oldest_detection = min(detections, key=lambda d: d['timestamp'])
                age = current_time - oldest_detection['timestamp']
                
                if age > timeout:
                    identities_to_process.append(reg_num)
            
            # Process timed-out buffers
            for reg_num in identities_to_process:
                detections = self.detection_buffer[reg_num]
                
                if len(detections) >= config.FUSION_MIN_DETECTIONS:
                    self._fuse_and_decide(reg_num)
                else:
                    # Not enough detections, discard
                    self.detection_buffer[reg_num].clear()
    
    def get_statistics(self) -> Dict:
        """Get fusion statistics"""
        with self.buffer_lock:
            active_buffers = sum(1 for d in self.detection_buffer.values() if d)
        
        stats = {
            'total_fused': self.total_fused,
            'total_decisions': self.total_decisions,
            'queue_size': self.detection_queue.qsize(),
            'active_buffers': active_buffers,
            'attendance_engine': self.attendance_engine.get_statistics()
        }
        
        return stats
    
    def get_today_summary(self) -> Dict:
        """Get today's attendance summary"""
        return self.attendance_engine.get_today_summary()
    
    def cleanup(self):
        """Cleanup old data"""
        self.attendance_engine.cleanup()


class SimpleFusionEngine:
    """
    Simplified fusion engine (direct pass-through)
    No aggregation, processes each detection immediately
    """
    
    def __init__(self):
        self.attendance_engine = AttendanceEngine()
        print("[SimpleFusionEngine] Initialized (pass-through mode)")
    
    def add_detection(self, camera_id: str, register_number: str,
                     confidence: float, location: str,
                     face_location: Dict = None):
        """Process detection immediately"""
        result = self.attendance_engine.process_detection(
            camera_id, register_number, confidence, location, face_location
        )
        return result
    
    def add_unknown_detection(self, camera_id: str, confidence: float,
                             face_location: Dict = None):
        """Process unknown detection"""
        self.attendance_engine.process_unknown_detection(
            camera_id, confidence, face_location
        )
    
    def start(self):
        """No-op for simple fusion"""
        pass
    
    def stop(self):
        """No-op for simple fusion"""
        pass
    
    def get_statistics(self) -> Dict:
        """Get statistics"""
        return self.attendance_engine.get_statistics()
    
    def get_today_summary(self) -> Dict:
        """Get today's summary"""
        return self.attendance_engine.get_today_summary()
