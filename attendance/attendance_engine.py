"""
Attendance Engine Module
Core attendance marking logic with event-based processing
Implements multi-frame validation and confidence aggregation
"""

import time
from datetime import datetime, date
from typing import Dict, Optional, List
import logging

import config
from database.db import get_db
from attendance.deduplicator import AttendanceDeduplicator, DetectionMerger
from attendance.confidence_tracker import ConfidenceTracker, TemporalValidator


class AttendanceEngine:
    """
    Central attendance marking engine
    Event-driven attendance system with validation
    """
    
    def __init__(self):
        self.db = get_db()
        
        # Core components
        self.deduplicator = AttendanceDeduplicator()
        self.confidence_tracker = ConfidenceTracker()
        self.temporal_validator = TemporalValidator()
        self.detection_merger = DetectionMerger()
        
        # Statistics
        self.total_detections = 0
        self.total_recognitions = 0
        self.total_attendance_marked = 0
        self.total_duplicates_prevented = 0
        
        # Setup logging
        self._setup_logging()
        
        print("[AttendanceEngine] Initialized")
        print(f"[AttendanceEngine] Min confidence: {config.ATTENDANCE_MIN_CONFIDENCE}")
        print(f"[AttendanceEngine] Cooldown: {config.ATTENDANCE_COOLDOWN}s")
    
    def _setup_logging(self):
        """Setup attendance logging"""
        self.logger = logging.getLogger('AttendanceEngine')
        self.logger.setLevel(logging.INFO)
        
        if config.LOG_TO_FILE:
            handler = logging.FileHandler(config.ATTENDANCE_LOG)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def process_detection(self, camera_id: str, register_number: str,
                         confidence: float, location: str,
                         face_location: Dict = None) -> Dict:
        """
        Process a single detection event
        
        Args:
            camera_id: Camera identifier
            register_number: Student register number
            confidence: Recognition confidence
            location: Camera location
            face_location: Face bounding box {x, y, w, h}
        
        Returns:
            Processing result dictionary
        """
        self.total_detections += 1
        timestamp = time.time()
        
        result = {
            'camera_id': camera_id,
            'register_number': register_number,
            'confidence': confidence,
            'timestamp': timestamp,
            'action': 'none',
            'reason': '',
            'attendance_marked': False
        }
        
        # Get student info
        student = self.db.get_student_by_register_number(register_number)
        if not student:
            result['action'] = 'rejected'
            result['reason'] = 'student_not_found'
            return result
        
        student_id = student['id']
        
        # Log detection event
        self.db.log_detection_event(
            camera_id, student_id, register_number,
            confidence, face_location or {}
        )
        
        # Add to confidence tracker
        self.confidence_tracker.add_detection(camera_id, register_number, confidence, timestamp)
        
        # Add to temporal validator
        self.temporal_validator.add_detection(register_number, timestamp)
        
        # Add to detection merger
        self.detection_merger.add_detection(register_number, camera_id, confidence, timestamp)
        
        # Check if confidence meets threshold
        if confidence < config.ATTENDANCE_MIN_CONFIDENCE:
            result['action'] = 'tracking'
            result['reason'] = f'low_confidence_{confidence:.2f}'
            return result
        
        self.total_recognitions += 1
        
        # Check if we have enough data to make decision
        if not config.ATTENDANCE_REQUIRES_MULTIPLE_FRAMES:
            # Single frame validation
            return self._attempt_mark_attendance(
                student_id, register_number, camera_id, location, confidence
            )
        
        # Multi-frame validation required
        if not self.confidence_tracker.should_make_decision(camera_id, register_number):
            result['action'] = 'accumulating'
            result['reason'] = 'collecting_more_samples'
            return result
        
        # Check temporal consistency
        if not self.temporal_validator.validate_detection(register_number):
            result['action'] = 'tracking'
            result['reason'] = 'temporal_validation_failed'
            return result
        
        # Get aggregated confidence
        agg_conf = self.confidence_tracker.get_aggregated_confidence(camera_id, register_number)
        
        if not agg_conf or not agg_conf['is_reliable']:
            result['action'] = 'rejected'
            result['reason'] = 'unreliable_confidence'
            return result
        
        # Attempt to mark attendance with aggregated confidence
        final_confidence = agg_conf['weighted_confidence']
        detection_count = agg_conf['sample_count']
        
        return self._attempt_mark_attendance(
            student_id, register_number, camera_id, location,
            final_confidence, detection_count
        )
    
    def _attempt_mark_attendance(self, student_id: int, register_number: str,
                                camera_id: str, location: str,
                                confidence: float, detection_count: int = 1) -> Dict:
        """
        Attempt to mark attendance
        
        Args:
            student_id: Student database ID
            register_number: Student register number
            camera_id: Camera identifier
            location: Camera location
            confidence: Final confidence score
            detection_count: Number of detections aggregated
        
        Returns:
            Marking result
        """
        result = {
            'register_number': register_number,
            'confidence': confidence,
            'detection_count': detection_count,
            'action': 'none',
            'reason': '',
            'attendance_marked': False
        }
        
        # Check deduplication
        can_mark, reason = self.deduplicator.can_mark_attendance(register_number)
        
        if not can_mark:
            result['action'] = 'duplicate_prevented'
            result['reason'] = reason
            self.total_duplicates_prevented += 1
            
            self.logger.info(
                f"Duplicate prevented: {register_number} - {reason}"
            )
            
            return result
        
        # Mark attendance attempt (starts cooldown)
        self.deduplicator.mark_attendance_attempt(register_number)
        
        # Mark in database
        success = self.db.mark_attendance(
            student_id, register_number, camera_id, location,
            confidence, detection_count
        )
        
        if success:
            result['action'] = 'marked'
            result['reason'] = 'success'
            result['attendance_marked'] = True
            
            # Update deduplicator
            self.deduplicator.mark_attendance_success(register_number)
            
            # Mark decision made in confidence tracker
            self.confidence_tracker.mark_decision_made(register_number)
            
            # Clear tracking data
            self.confidence_tracker.clear_identity(register_number)
            self.temporal_validator.clear_identity(register_number)
            self.detection_merger.clear_identity(register_number)
            
            self.total_attendance_marked += 1
            
            self.logger.info(
                f"Attendance marked: {register_number} | "
                f"Camera: {camera_id} | Confidence: {confidence:.3f} | "
                f"Detections: {detection_count}"
            )
            
            print(f"\n✓ ATTENDANCE MARKED: {register_number} (confidence: {confidence:.3f})")
        else:
            result['action'] = 'failed'
            result['reason'] = 'database_error'
            
            self.logger.error(
                f"Failed to mark attendance: {register_number}"
            )
        
        return result
    
    def process_unknown_detection(self, camera_id: str, confidence: float,
                                  face_location: Dict = None):
        """
        Process detection of unknown face
        
        Args:
            camera_id: Camera identifier
            confidence: Detection confidence
            face_location: Face bounding box
        """
        self.total_detections += 1
        
        # Log unknown detection
        self.db.log_unknown_detection(camera_id, confidence, face_location or {})
        
        self.logger.debug(
            f"Unknown face detected: Camera {camera_id} | "
            f"Confidence: {confidence:.3f}"
        )
    
    def get_today_summary(self) -> Dict:
        """Get today's attendance summary"""
        attendance_records = self.db.get_today_attendance()
        
        if not attendance_records:
            return {
                'date': date.today().isoformat(),
                'total_present': 0,
                'records': []
            }
        
        summary = {
            'date': date.today().isoformat(),
            'total_present': len(attendance_records),
            'avg_confidence': sum(r['confidence_score'] for r in attendance_records) / len(attendance_records),
            'records': attendance_records
        }
        
        return summary
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        return {
            'total_detections': self.total_detections,
            'total_recognitions': self.total_recognitions,
            'total_attendance_marked': self.total_attendance_marked,
            'total_duplicates_prevented': self.total_duplicates_prevented,
            'recognition_rate': (
                self.total_recognitions / self.total_detections 
                if self.total_detections > 0 else 0
            ),
            'attendance_rate': (
                self.total_attendance_marked / self.total_recognitions 
                if self.total_recognitions > 0 else 0
            ),
            'deduplicator': self.deduplicator.get_statistics(),
            'confidence_tracker': self.confidence_tracker.get_statistics()
        }
    
    def cleanup(self):
        """Cleanup old tracking data"""
        self.confidence_tracker.cleanup_old_data()
        self.detection_merger.cleanup_old_detections()
    
    def reset_daily(self):
        """Reset daily tracking (call at midnight)"""
        print("[AttendanceEngine] Performing daily reset")
        self.deduplicator.reset_daily()
        
        # Reset statistics
        self.total_detections = 0
        self.total_recognitions = 0
        self.total_attendance_marked = 0
        self.total_duplicates_prevented = 0
