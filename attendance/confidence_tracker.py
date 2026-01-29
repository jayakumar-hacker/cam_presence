"""
Confidence Tracker Module
Tracks recognition confidence over time windows
Implements temporal aggregation for decision making
"""

import time
from collections import deque, defaultdict
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import numpy as np

import config


class ConfidenceTracker:
    """
    Tracks confidence scores over time windows
    Aggregates multiple detections for robust decision making
    """
    
    def __init__(self):
        # Track confidence samples per identity per camera
        # Structure: {(camera_id, register_number): deque of (timestamp, confidence)}
        self.confidence_windows = defaultdict(lambda: deque(maxlen=config.CONFIDENCE_WINDOW_SIZE))
        
        # Track last decision time per identity
        self.last_decision_time = {}
        
        print("[ConfidenceTracker] Initialized")
        print(f"[ConfidenceTracker] Window size: {config.CONFIDENCE_WINDOW_SIZE}")
        print(f"[ConfidenceTracker] Min samples: {config.CONFIDENCE_MIN_SAMPLES}")
    
    def add_detection(self, camera_id: str, register_number: str, 
                     confidence: float, timestamp: float = None):
        """
        Add a detection event with confidence score
        
        Args:
            camera_id: Camera identifier
            register_number: Student register number
            confidence: Recognition confidence (0-1)
            timestamp: Detection timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        key = (camera_id, register_number)
        self.confidence_windows[key].append((timestamp, confidence))
    
    def get_aggregated_confidence(self, camera_id: str, 
                                  register_number: str) -> Optional[Dict]:
        """
        Get aggregated confidence for an identity
        
        Args:
            camera_id: Camera identifier
            register_number: Student register number
        
        Returns:
            Dictionary with aggregated metrics or None
        """
        key = (camera_id, register_number)
        
        if key not in self.confidence_windows:
            return None
        
        samples = list(self.confidence_windows[key])
        
        if len(samples) < config.CONFIDENCE_MIN_SAMPLES:
            return None
        
        # Extract confidence values
        confidences = [conf for _, conf in samples]
        
        # Calculate statistics
        mean_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        std_confidence = np.std(confidences)
        
        # Calculate weighted confidence (recent samples weighted more)
        weights = np.linspace(0.5, 1.0, len(confidences))
        weighted_confidence = np.average(confidences, weights=weights)
        
        # Time span
        timestamps = [ts for ts, _ in samples]
        time_span = max(timestamps) - min(timestamps)
        
        return {
            'camera_id': camera_id,
            'register_number': register_number,
            'sample_count': len(samples),
            'mean_confidence': mean_confidence,
            'weighted_confidence': weighted_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'std_confidence': std_confidence,
            'time_span': time_span,
            'is_stable': std_confidence < 0.1,  # Low variance = stable
            'is_reliable': mean_confidence > config.ATTENDANCE_MIN_CONFIDENCE
        }
    
    def get_multi_camera_confidence(self, register_number: str) -> Optional[Dict]:
        """
        Get aggregated confidence across all cameras for an identity
        
        Args:
            register_number: Student register number
        
        Returns:
            Multi-camera aggregated metrics
        """
        # Get all cameras that have seen this identity
        camera_data = []
        
        for (cam_id, reg_num), samples in self.confidence_windows.items():
            if reg_num == register_number:
                if len(samples) >= config.CONFIDENCE_MIN_SAMPLES:
                    confidences = [conf for _, conf in samples]
                    camera_data.append({
                        'camera_id': cam_id,
                        'mean_confidence': np.mean(confidences),
                        'sample_count': len(samples)
                    })
        
        if not camera_data:
            return None
        
        # Aggregate across cameras
        all_confidences = [data['mean_confidence'] for data in camera_data]
        total_samples = sum(data['sample_count'] for data in camera_data)
        
        return {
            'register_number': register_number,
            'camera_count': len(camera_data),
            'cameras': camera_data,
            'overall_confidence': np.mean(all_confidences),
            'max_camera_confidence': np.max(all_confidences),
            'total_samples': total_samples,
            'is_multi_camera': len(camera_data) > 1
        }
    
    def should_make_decision(self, camera_id: str, register_number: str) -> bool:
        """
        Check if enough data collected to make a decision
        
        Args:
            camera_id: Camera identifier
            register_number: Student register number
        
        Returns:
            True if ready to make decision
        """
        agg_conf = self.get_aggregated_confidence(camera_id, register_number)
        
        if agg_conf is None:
            return False
        
        # Check if we have enough samples
        if agg_conf['sample_count'] < config.CONFIDENCE_MIN_SAMPLES:
            return False
        
        # Check if confidence is reliable
        if not agg_conf['is_reliable']:
            return False
        
        # Check cooldown period
        if register_number in self.last_decision_time:
            elapsed = time.time() - self.last_decision_time[register_number]
            if elapsed < config.ATTENDANCE_COOLDOWN:
                return False
        
        return True
    
    def mark_decision_made(self, register_number: str):
        """
        Mark that a decision was made for this identity
        Starts cooldown period
        
        Args:
            register_number: Student register number
        """
        self.last_decision_time[register_number] = time.time()
    
    def clear_identity(self, register_number: str):
        """
        Clear all tracking data for an identity
        
        Args:
            register_number: Student register number
        """
        keys_to_remove = []
        
        for key in self.confidence_windows.keys():
            cam_id, reg_num = key
            if reg_num == register_number:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.confidence_windows[key]
    
    def cleanup_old_data(self, max_age: float = 60.0):
        """
        Remove old tracking data
        
        Args:
            max_age: Maximum age in seconds
        """
        current_time = time.time()
        
        for key, samples in list(self.confidence_windows.items()):
            # Filter out old samples
            fresh_samples = deque(
                [(ts, conf) for ts, conf in samples 
                 if current_time - ts < max_age],
                maxlen=config.CONFIDENCE_WINDOW_SIZE
            )
            
            if len(fresh_samples) == 0:
                del self.confidence_windows[key]
            else:
                self.confidence_windows[key] = fresh_samples
    
    def get_statistics(self) -> Dict:
        """Get tracker statistics"""
        active_identities = set()
        total_samples = 0
        
        for (cam_id, reg_num), samples in self.confidence_windows.items():
            active_identities.add(reg_num)
            total_samples += len(samples)
        
        return {
            'active_tracking': len(self.confidence_windows),
            'unique_identities': len(active_identities),
            'total_samples': total_samples,
            'avg_samples_per_identity': (
                total_samples / len(active_identities) 
                if active_identities else 0
            )
        }


class TemporalValidator:
    """
    Validates detections across time
    Ensures temporal consistency before marking attendance
    """
    
    def __init__(self):
        # Track detection events: {register_number: [timestamps]}
        self.detection_history = defaultdict(list)
        
        print("[TemporalValidator] Initialized")
    
    def add_detection(self, register_number: str, timestamp: float = None):
        """
        Add detection event
        
        Args:
            register_number: Student register number
            timestamp: Detection timestamp
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.detection_history[register_number].append(timestamp)
        
        # Keep only recent history
        self._cleanup_old_detections(register_number)
    
    def validate_detection(self, register_number: str) -> bool:
        """
        Validate if detections are temporally consistent
        
        Args:
            register_number: Student register number
        
        Returns:
            True if detection pattern is valid
        """
        if register_number not in self.detection_history:
            return False
        
        detections = self.detection_history[register_number]
        
        # Need minimum number of detections
        if len(detections) < config.ATTENDANCE_MIN_VALIDATION_FRAMES:
            return False
        
        # Check time span
        if len(detections) >= 2:
            time_span = max(detections) - min(detections)
            
            # Detections should span at least a few seconds
            if time_span < 2.0:
                return False
            
            # But not too long (person should be present continuously)
            if time_span > 30.0:
                return False
        
        return True
    
    def _cleanup_old_detections(self, register_number: str, max_age: float = 60.0):
        """Remove old detections"""
        if register_number not in self.detection_history:
            return
        
        current_time = time.time()
        fresh_detections = [
            ts for ts in self.detection_history[register_number]
            if current_time - ts < max_age
        ]
        
        if fresh_detections:
            self.detection_history[register_number] = fresh_detections
        else:
            del self.detection_history[register_number]
    
    def clear_identity(self, register_number: str):
        """Clear history for identity"""
        if register_number in self.detection_history:
            del self.detection_history[register_number]
