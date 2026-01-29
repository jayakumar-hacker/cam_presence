"""
RTSP/HTTP Stream Handler Module
Manages video stream connections from IP cameras
Handles reconnection, frame buffering, and error recovery
"""

import cv2
import time
import threading
from queue import Queue, Full
from typing import Optional, Tuple
import numpy as np

import config


class StreamHandler:
    """
    Handles video stream from IP camera (RTSP/HTTP)
    Implements reconnection logic and frame buffering
    """
    
    def __init__(self, camera_id: str, stream_url: str, location: str):
        self.camera_id = camera_id
        self.stream_url = stream_url
        self.location = location
        
        self.capture = None
        self.is_running = False
        self.is_connected = False
        
        # Frame management
        self.frame_queue = Queue(maxsize=config.FRAME_BUFFER_SIZE)
        self.current_frame = None
        self.frame_count = 0
        self.last_frame_time = None
        
        # Threading
        self.read_thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.total_frames = 0
        self.dropped_frames = 0
        self.reconnect_count = 0
        
        print(f"[{self.camera_id}] Stream handler initialized for {stream_url}")
    
    def connect(self) -> bool:
        """Connect to video stream"""
        try:
            print(f"[{self.camera_id}] Connecting to {self.stream_url}...")
            
            # Try to open video capture
            self.capture = cv2.VideoCapture(self.stream_url)
            
            if not self.capture.isOpened():
                print(f"[{self.camera_id}] Failed to open stream")
                return False
            
            # Test read a frame
            ret, frame = self.capture.read()
            if not ret:
                print(f"[{self.camera_id}] Failed to read initial frame")
                self.capture.release()
                return False
            
            self.is_connected = True
            self.last_frame_time = time.time()
            
            print(f"[{self.camera_id}] Successfully connected")
            print(f"[{self.camera_id}] Frame size: {frame.shape[1]}x{frame.shape[0]}")
            
            return True
            
        except Exception as e:
            print(f"[{self.camera_id}] Connection error: {e}")
            return False
    
    def start(self):
        """Start streaming in separate thread"""
        if self.is_running:
            print(f"[{self.camera_id}] Already running")
            return
        
        if not self.connect():
            print(f"[{self.camera_id}] Cannot start - connection failed")
            return
        
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        
        print(f"[{self.camera_id}] Stream started")
    
    def _read_loop(self):
        """Main loop for reading frames from stream"""
        frame_skip_counter = 0
        
        while self.is_running:
            try:
                if not self.is_connected:
                    # Try to reconnect
                    print(f"[{self.camera_id}] Attempting reconnection...")
                    if self.connect():
                        self.reconnect_count += 1
                        print(f"[{self.camera_id}] Reconnected (attempt #{self.reconnect_count})")
                    else:
                        time.sleep(config.STREAM_RECONNECT_DELAY)
                        continue
                
                # Read frame
                ret, frame = self.capture.read()
                
                if not ret:
                    print(f"[{self.camera_id}] Frame read failed")
                    self.is_connected = False
                    if self.capture:
                        self.capture.release()
                    continue
                
                # Update statistics
                self.total_frames += 1
                self.frame_count += 1
                self.last_frame_time = time.time()
                
                # Frame skipping for performance
                frame_skip_counter += 1
                if frame_skip_counter % config.FRAME_SKIP != 0:
                    continue
                
                # Store current frame
                with self.lock:
                    self.current_frame = frame.copy()
                
                # Try to add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    self.dropped_frames += 1
                    # Queue is full, drop oldest frame
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
                
            except Exception as e:
                print(f"[{self.camera_id}] Read loop error: {e}")
                self.is_connected = False
                time.sleep(1)
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get latest frame from queue (blocking with timeout)"""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except:
            return None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current frame (non-blocking)"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def is_alive(self) -> bool:
        """Check if stream is alive and receiving frames"""
        if not self.is_connected or not self.is_running:
            return False
        
        if self.last_frame_time is None:
            return False
        
        # Check if we received a frame recently
        time_since_last_frame = time.time() - self.last_frame_time
        return time_since_last_frame < config.STREAM_READ_TIMEOUT
    
    def get_stats(self) -> dict:
        """Get stream statistics"""
        return {
            'camera_id': self.camera_id,
            'location': self.location,
            'is_connected': self.is_connected,
            'is_alive': self.is_alive(),
            'total_frames': self.total_frames,
            'dropped_frames': self.dropped_frames,
            'reconnect_count': self.reconnect_count,
            'queue_size': self.frame_queue.qsize(),
            'last_frame_time': self.last_frame_time
        }
    
    def stop(self):
        """Stop streaming"""
        print(f"[{self.camera_id}] Stopping stream...")
        
        self.is_running = False
        
        if self.read_thread:
            self.read_thread.join(timeout=2)
        
        if self.capture:
            self.capture.release()
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        print(f"[{self.camera_id}] Stream stopped")
        print(f"[{self.camera_id}] Total frames: {self.total_frames}, Dropped: {self.dropped_frames}")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()


class MultiStreamManager:
    """
    Manages multiple camera streams
    Central coordinator for all video sources
    """
    
    def __init__(self):
        self.streams = {}
        self.is_running = False
        
        print("[StreamManager] Initialized")
    
    def add_camera(self, camera_config: dict):
        """Add a camera stream"""
        camera_id = camera_config['camera_id']
        stream_url = camera_config['stream_url']
        location = camera_config['location']
        
        if camera_id in self.streams:
            print(f"[StreamManager] Camera {camera_id} already exists")
            return
        
        handler = StreamHandler(camera_id, stream_url, location)
        self.streams[camera_id] = handler
        
        print(f"[StreamManager] Added camera: {camera_id} ({location})")
    
    def start_all(self):
        """Start all camera streams"""
        print("[StreamManager] Starting all cameras...")
        
        self.is_running = True
        
        for camera_id, handler in self.streams.items():
            if not handler.is_running:
                handler.start()
                time.sleep(1)  # Stagger camera starts
        
        print(f"[StreamManager] Started {len(self.streams)} cameras")
    
    def stop_all(self):
        """Stop all camera streams"""
        print("[StreamManager] Stopping all cameras...")
        
        self.is_running = False
        
        for camera_id, handler in self.streams.items():
            handler.stop()
        
        print("[StreamManager] All cameras stopped")
    
    def get_stream(self, camera_id: str) -> Optional[StreamHandler]:
        """Get stream handler by camera ID"""
        return self.streams.get(camera_id)
    
    def get_all_streams(self) -> dict:
        """Get all stream handlers"""
        return self.streams
    
    def get_active_cameras(self) -> list:
        """Get list of active camera IDs"""
        return [
            camera_id for camera_id, handler in self.streams.items()
            if handler.is_alive()
        ]
    
    def get_system_stats(self) -> dict:
        """Get statistics for all cameras"""
        stats = {
            'total_cameras': len(self.streams),
            'active_cameras': len(self.get_active_cameras()),
            'cameras': []
        }
        
        for camera_id, handler in self.streams.items():
            stats['cameras'].append(handler.get_stats())
        
        return stats
    
    def health_check(self):
        """Perform health check on all streams"""
        print("\n[StreamManager] Health Check:")
        print("=" * 60)
        
        for camera_id, handler in self.streams.items():
            status = "🟢 ALIVE" if handler.is_alive() else "🔴 DEAD"
            stats = handler.get_stats()
            
            print(f"Camera: {camera_id} ({handler.location})")
            print(f"  Status: {status}")
            print(f"  Total Frames: {stats['total_frames']}")
            print(f"  Dropped Frames: {stats['dropped_frames']}")
            print(f"  Reconnections: {stats['reconnect_count']}")
            print(f"  Queue Size: {stats['queue_size']}")
            print()
        
        print("=" * 60)
