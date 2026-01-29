"""
Database Handler Module
Manages all database operations for the attendance system
Thread-safe SQLite operations with connection pooling
"""

import sqlite3
import threading
import pickle
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

import config


class DatabaseManager:
    """
    Thread-safe database manager for SQLite operations
    Implements connection pooling and safe concurrent access
    """
    
    def __init__(self, db_path: Path = config.DATABASE_PATH):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._local = threading.local()
        self._initialize_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                timeout=config.DB_TIMEOUT,
                check_same_thread=False
            )
            self._local.connection.row_factory = sqlite3.Row
        return self._local.connection
    
    def _initialize_database(self):
        """Initialize database with schema"""
        schema_path = Path(__file__).parent / "schema.sql"
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(schema_sql)
        conn.commit()
        conn.close()
        
        print(f"[DB] Database initialized at {self.db_path}")
    
    # ===========================
    # STUDENT OPERATIONS
    # ===========================
    
    def register_student(self, register_number: str, name: str, 
                        email: str = None, department: str = None, 
                        year: int = None) -> int:
        """Register a new student"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO students (register_number, name, email, department, year)
                    VALUES (?, ?, ?, ?, ?)
                """, (register_number, name, email, department, year))
                
                conn.commit()
                student_id = cursor.lastrowid
                print(f"[DB] Student registered: {name} ({register_number})")
                return student_id
                
            except sqlite3.IntegrityError:
                print(f"[DB] Student already exists: {register_number}")
                cursor.execute(
                    "SELECT id FROM students WHERE register_number = ?",
                    (register_number,)
                )
                return cursor.fetchone()[0]
    
    def get_student_by_register_number(self, register_number: str) -> Optional[Dict]:
        """Get student details by register number"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM students WHERE register_number = ? AND is_active = 1",
            (register_number,)
        )
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_all_students(self) -> List[Dict]:
        """Get all active students"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM students WHERE is_active = 1 ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]
    
    # ===========================
    # EMBEDDING OPERATIONS
    # ===========================
    
    def store_embedding(self, student_id: int, embedding: np.ndarray, 
                       quality: float, camera_id: str = None) -> int:
        """Store face embedding for a student"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Serialize embedding as bytes
            embedding_bytes = pickle.dumps(embedding)
            
            cursor.execute("""
                INSERT INTO face_embeddings 
                (student_id, embedding_vector, embedding_quality, camera_id)
                VALUES (?, ?, ?, ?)
            """, (student_id, embedding_bytes, quality, camera_id))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_all_embeddings(self) -> List[Dict]:
        """Get all face embeddings with student information"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                fe.id, fe.student_id, fe.embedding_vector, 
                fe.embedding_quality, s.register_number, s.name
            FROM face_embeddings fe
            JOIN students s ON fe.student_id = s.id
            WHERE s.is_active = 1
        """)
        
        results = []
        for row in cursor.fetchall():
            row_dict = dict(row)
            # Deserialize embedding
            row_dict['embedding_vector'] = pickle.loads(row_dict['embedding_vector'])
            results.append(row_dict)
        
        return results
    
    def get_student_embeddings(self, student_id: int) -> List[np.ndarray]:
        """Get all embeddings for a specific student"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT embedding_vector FROM face_embeddings WHERE student_id = ?",
            (student_id,)
        )
        
        embeddings = []
        for row in cursor.fetchall():
            embedding = pickle.loads(row[0])
            embeddings.append(embedding)
        
        return embeddings
    
    # ===========================
    # ATTENDANCE OPERATIONS
    # ===========================
    
    def mark_attendance(self, student_id: int, register_number: str,
                       camera_id: str, location: str, 
                       confidence: float, detection_count: int = 1) -> bool:
        """Mark attendance for a student (event-based)"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            today = date.today()
            now = datetime.now()
            
            try:
                # Check if already marked today
                cursor.execute("""
                    SELECT id FROM attendance_records 
                    WHERE student_id = ? AND attendance_date = ?
                """, (student_id, today))
                
                if cursor.fetchone():
                    print(f"[DB] Attendance already marked for {register_number} today")
                    return False
                
                # Mark new attendance
                cursor.execute("""
                    INSERT INTO attendance_records 
                    (student_id, register_number, attendance_date, check_in_time,
                     camera_id, location, confidence_score, detection_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (student_id, register_number, today, now, 
                      camera_id, location, confidence, detection_count))
                
                conn.commit()
                print(f"[DB] Attendance marked: {register_number} at {now.strftime('%H:%M:%S')}")
                return True
                
            except Exception as e:
                print(f"[DB] Error marking attendance: {e}")
                conn.rollback()
                return False
    
    def get_today_attendance(self) -> List[Dict]:
        """Get all attendance records for today"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = date.today()
        
        cursor.execute("""
            SELECT ar.*, s.name 
            FROM attendance_records ar
            JOIN students s ON ar.student_id = s.id
            WHERE ar.attendance_date = ?
            ORDER BY ar.check_in_time
        """, (today,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def check_attendance_today(self, student_id: int) -> bool:
        """Check if student attendance is already marked today"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        today = date.today()
        
        cursor.execute("""
            SELECT id FROM attendance_records 
            WHERE student_id = ? AND attendance_date = ?
        """, (student_id, today))
        
        return cursor.fetchone() is not None
    
    # ===========================
    # DETECTION EVENT OPERATIONS
    # ===========================
    
    def log_detection_event(self, camera_id: str, student_id: int,
                           register_number: str, confidence: float,
                           face_location: Dict, frame_number: int = None) -> int:
        """Log a detection event"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = datetime.now()
            face_location_json = json.dumps(face_location)
            
            cursor.execute("""
                INSERT INTO detection_events
                (camera_id, student_id, register_number, detection_time,
                 confidence_score, face_location, is_recognized, frame_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (camera_id, student_id, register_number, now,
                  confidence, face_location_json, True, frame_number))
            
            conn.commit()
            return cursor.lastrowid
    
    def log_unknown_detection(self, camera_id: str, confidence: float,
                             face_location: Dict, frame_number: int = None) -> int:
        """Log detection of unknown face"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = datetime.now()
            face_location_json = json.dumps(face_location)
            
            cursor.execute("""
                INSERT INTO detection_events
                (camera_id, detection_time, confidence_score, 
                 face_location, is_recognized, frame_number)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (camera_id, now, confidence, face_location_json, False, frame_number))
            
            conn.commit()
            return cursor.lastrowid
    
    # ===========================
    # FUSION EVENT OPERATIONS
    # ===========================
    
    def log_fusion_event(self, student_id: int, register_number: str,
                        total_detections: int, camera_count: int,
                        avg_confidence: float, max_confidence: float,
                        cameras: List[str], decision: str) -> int:
        """Log a fusion decision event"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = datetime.now()
            cameras_json = json.dumps(cameras)
            
            cursor.execute("""
                INSERT INTO fusion_events
                (student_id, register_number, event_time, total_detections,
                 camera_count, average_confidence, max_confidence, 
                 cameras_detected, decision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (student_id, register_number, now, total_detections,
                  camera_count, avg_confidence, max_confidence,
                  cameras_json, decision))
            
            conn.commit()
            return cursor.lastrowid
    
    # ===========================
    # CONFIDENCE TRACKING OPERATIONS
    # ===========================
    
    def log_confidence_sample(self, student_id: int, register_number: str,
                             camera_id: str, confidence: float,
                             window_start: datetime, window_end: datetime) -> int:
        """Log confidence tracking sample"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = datetime.now()
            
            cursor.execute("""
                INSERT INTO confidence_tracking
                (student_id, register_number, camera_id, confidence_score,
                 detection_time, window_start, window_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (student_id, register_number, camera_id, confidence,
                  now, window_start, window_end))
            
            conn.commit()
            return cursor.lastrowid
    
    # ===========================
    # CAMERA STATUS OPERATIONS
    # ===========================
    
    def update_camera_status(self, camera_id: str, status: str,
                            location: str = None):
        """Update camera status"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            now = datetime.now()
            
            cursor.execute("""
                INSERT INTO camera_status (camera_id, location, status, last_frame_time, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(camera_id) DO UPDATE SET
                    status = excluded.status,
                    last_frame_time = excluded.last_frame_time,
                    updated_at = excluded.updated_at
            """, (camera_id, location, status, now, now))
            
            conn.commit()
    
    def increment_camera_stats(self, camera_id: str, frames: int = 0, detections: int = 0):
        """Increment camera statistics"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE camera_status
                SET total_frames_processed = total_frames_processed + ?,
                    total_detections = total_detections + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE camera_id = ?
            """, (frames, detections, camera_id))
            
            conn.commit()
    
    # ===========================
    # SYSTEM LOG OPERATIONS
    # ===========================
    
    def log_system_event(self, level: str, component: str, 
                        message: str, details: str = None):
        """Log system event"""
        with self.lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO system_logs (log_level, component, message, details)
                VALUES (?, ?, ?, ?)
            """, (level, component, message, details))
            
            conn.commit()
    
    # ===========================
    # UTILITY OPERATIONS
    # ===========================
    
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total students
        cursor.execute("SELECT COUNT(*) FROM students WHERE is_active = 1")
        stats['total_students'] = cursor.fetchone()[0]
        
        # Total embeddings
        cursor.execute("SELECT COUNT(*) FROM face_embeddings")
        stats['total_embeddings'] = cursor.fetchone()[0]
        
        # Today's attendance
        today = date.today()
        cursor.execute(
            "SELECT COUNT(*) FROM attendance_records WHERE attendance_date = ?",
            (today,)
        )
        stats['today_attendance'] = cursor.fetchone()[0]
        
        # Total detection events
        cursor.execute("SELECT COUNT(*) FROM detection_events")
        stats['total_detections'] = cursor.fetchone()[0]
        
        return stats
    
    def close(self):
        """Close database connection"""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()


# Singleton instance
_db_instance = None

def get_db() -> DatabaseManager:
    """Get singleton database instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
