"""
Deduplication Module
Identity-based deduplication for attendance marking
Prevents duplicate attendance entries
"""

import time
from collections import defaultdict
from typing import Dict, Optional, Set
from datetime import datetime, date

import config
from database.db import get_db


class AttendanceDeduplicator:
    """
    Deduplicates attendance marking events
    Ensures one attendance per identity per day
    Implements cooldown and session management
    """
    
    def __init__(self):
        self.db = get_db()
        
        # In-memory tracking for current session
        # {register_number: last_marked_timestamp}
        self.marked_today = {}
        
        # Cooldown tracking: {register_number: last_detection_timestamp}
        self.cooldown_tracker = {}
        
        # Load today's attendance
        self._load_today_attendance()
        
        print("[Deduplicator] Initialized")
        print(f"[Deduplicator] Cooldown period: {config.ATTENDANCE_COOLDOWN}s")
    
    def _load_today_attendance(self):
        """Load attendance already marked today"""
        today_records = self.db.get_today_attendance()
        
        for record in today_records:
            register_number = record['register_number']
            check_in_time = record['check_in_time']
            
            # Parse timestamp
            if isinstance(check_in_time, str):
                check_in_time = datetime.fromisoformat(check_in_time)
            
            self.marked_today[register_number] = check_in_time.timestamp()
        
        print(f"[Deduplicator] Loaded {len(self.marked_today)} attendance records for today")
    
    def is_duplicate(self, register_number: str) -> bool:
        """
        Check if attendance would be duplicate
        
        Args:
            register_number: Student register number
        
        Returns:
            True if duplicate
        """
        # Check if already marked today
        if register_number in self.marked_today:
            return True
        
        # Double-check with database
        # (in case of system restart or multi-instance)
        student = self.db.get_student_by_register_number(register_number)
        if student:
            already_marked = self.db.check_attendance_today(student['id'])
            if already_marked:
                # Update local cache
                self.marked_today[register_number] = time.time()
                return True
        
        return False
    
    def is_in_cooldown(self, register_number: str) -> bool:
        """
        Check if identity is in cooldown period
        
        Args:
            register_number: Student register number
        
        Returns:
            True if in cooldown
        """
        if register_number not in self.cooldown_tracker:
            return False
        
        last_detection = self.cooldown_tracker[register_number]
        elapsed = time.time() - last_detection
        
        return elapsed < config.ATTENDANCE_COOLDOWN
    
    def can_mark_attendance(self, register_number: str) -> tuple:
        """
        Check if attendance can be marked
        
        Args:
            register_number: Student register number
        
        Returns:
            (can_mark: bool, reason: str)
        """
        # Check if duplicate
        if self.is_duplicate(register_number):
            return False, "already_marked_today"
        
        # Check cooldown
        if self.is_in_cooldown(register_number):
            remaining = config.ATTENDANCE_COOLDOWN - (
                time.time() - self.cooldown_tracker[register_number]
            )
            return False, f"cooldown_{int(remaining)}s"
        
        return True, "ok"
    
    def mark_attendance_attempt(self, register_number: str):
        """
        Record an attendance marking attempt
        Starts cooldown period
        
        Args:
            register_number: Student register number
        """
        current_time = time.time()
        self.cooldown_tracker[register_number] = current_time
    
    def mark_attendance_success(self, register_number: str):
        """
        Record successful attendance marking
        
        Args:
            register_number: Student register number
        """
        current_time = time.time()
        self.marked_today[register_number] = current_time
        self.cooldown_tracker[register_number] = current_time
    
    def reset_daily(self):
        """Reset daily tracking (call at midnight)"""
        print("[Deduplicator] Resetting daily attendance tracking")
        self.marked_today.clear()
        self.cooldown_tracker.clear()
        self._load_today_attendance()
    
    def get_marked_count(self) -> int:
        """Get number of students marked today"""
        return len(self.marked_today)
    
    def get_statistics(self) -> Dict:
        """Get deduplicator statistics"""
        return {
            'marked_today': len(self.marked_today),
            'in_cooldown': len(self.cooldown_tracker),
            'cooldown_period': config.ATTENDANCE_COOLDOWN
        }


class DetectionMerger:
    """
    Merges detection events within time windows
    Combines multiple detections of same identity
    """
    
    def __init__(self):
        # Detection buffer: {register_number: [detection_events]}
        self.detection_buffer = defaultdict(list)
        
        print("[DetectionMerger] Initialized")
        print(f"[DetectionMerger] Merge window: {config.DEDUP_TIME_WINDOW}s")
    
    def add_detection(self, register_number: str, camera_id: str,
                     confidence: float, timestamp: float = None) -> bool:
        """
        Add detection event to buffer
        
        Args:
            register_number: Student register number
            camera_id: Camera identifier
            confidence: Recognition confidence
            timestamp: Detection timestamp
        
        Returns:
            True if detection was merged, False if new
        """
        if timestamp is None:
            timestamp = time.time()
        
        detection = {
            'camera_id': camera_id,
            'confidence': confidence,
            'timestamp': timestamp
        }
        
        # Check if can be merged with existing detections
        if register_number in self.detection_buffer:
            last_detection = self.detection_buffer[register_number][-1]
            time_diff = timestamp - last_detection['timestamp']
            
            if time_diff < config.DEDUP_TIME_WINDOW:
                # Merge with existing
                self.detection_buffer[register_number].append(detection)
                return True
        
        # Start new detection sequence
        self.detection_buffer[register_number] = [detection]
        return False
    
    def get_merged_detection(self, register_number: str) -> Optional[Dict]:
        """
        Get merged detection for identity
        
        Args:
            register_number: Student register number
        
        Returns:
            Merged detection data or None
        """
        if register_number not in self.detection_buffer:
            return None
        
        detections = self.detection_buffer[register_number]
        
        if not detections:
            return None
        
        # Calculate merged statistics
        cameras = set(d['camera_id'] for d in detections)
        confidences = [d['confidence'] for d in detections]
        timestamps = [d['timestamp'] for d in detections]
        
        return {
            'register_number': register_number,
            'detection_count': len(detections),
            'cameras': list(cameras),
            'camera_count': len(cameras),
            'avg_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'first_seen': min(timestamps),
            'last_seen': max(timestamps),
            'time_span': max(timestamps) - min(timestamps)
        }
    
    def clear_identity(self, register_number: str):
        """Clear detection buffer for identity"""
        if register_number in self.detection_buffer:
            del self.detection_buffer[register_number]
    
    def cleanup_old_detections(self, max_age: float = 60.0):
        """Remove old detection buffers"""
        current_time = time.time()
        
        identities_to_remove = []
        
        for reg_num, detections in self.detection_buffer.items():
            if detections:
                last_detection_time = detections[-1]['timestamp']
                if current_time - last_detection_time > max_age:
                    identities_to_remove.append(reg_num)
        
        for reg_num in identities_to_remove:
            del self.detection_buffer[reg_num]


class SpatialDeduplicator:
    """
    Deduplicates based on spatial proximity
    Merges detections of same person from nearby locations
    """
    
    def __init__(self):
        # Track last known location: {register_number: (x, y, timestamp)}
        self.last_locations = {}
        
        print("[SpatialDeduplicator] Initialized")
        print(f"[SpatialDeduplicator] Spatial threshold: {config.DEDUP_SPATIAL_THRESHOLD}px")
    
    def update_location(self, register_number: str, x: int, y: int, 
                       timestamp: float = None):
        """
        Update last known location for identity
        
        Args:
            register_number: Student register number
            x: X coordinate (face center)
            y: Y coordinate (face center)
            timestamp: Detection timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.last_locations[register_number] = (x, y, timestamp)
    
    def is_same_location(self, register_number: str, x: int, y: int) -> bool:
        """
        Check if detection is at same location
        
        Args:
            register_number: Student register number
            x: X coordinate
            y: Y coordinate
        
        Returns:
            True if at same location (within threshold)
        """
        if register_number not in self.last_locations:
            return False
        
        last_x, last_y, last_time = self.last_locations[register_number]
        
        # Check time - if too old, consider different
        if time.time() - last_time > config.DEDUP_TIME_WINDOW:
            return False
        
        # Calculate Euclidean distance
        distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
        
        return distance < config.DEDUP_SPATIAL_THRESHOLD
    
    def clear_identity(self, register_number: str):
        """Clear location tracking for identity"""
        if register_number in self.last_locations:
            del self.last_locations[register_number]
