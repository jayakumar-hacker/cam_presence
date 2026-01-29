-- Smart CCTV Attendance System Database Schema
-- SQLite database for storing students, embeddings, and attendance records

-- Students table: Core identity information
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    register_number VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    department VARCHAR(100),
    year INTEGER,
    enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face embeddings table: Stores face embedding vectors
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    embedding_vector BLOB NOT NULL,  -- Serialized numpy array
    embedding_quality REAL,
    capture_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    camera_id VARCHAR(50),
    is_primary BOOLEAN DEFAULT 0,  -- Mark best quality embedding
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

-- Attendance records table: Event-based attendance marking
CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    register_number VARCHAR(50) NOT NULL,
    attendance_date DATE NOT NULL,
    check_in_time TIMESTAMP NOT NULL,
    check_out_time TIMESTAMP,
    camera_id VARCHAR(50),
    location VARCHAR(255),
    confidence_score REAL,
    detection_count INTEGER DEFAULT 1,
    status VARCHAR(20) DEFAULT 'present',  -- present, late, absent
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    UNIQUE(student_id, attendance_date)  -- One attendance per day per student
);

-- Detection events table: Raw detection events from cameras
CREATE TABLE IF NOT EXISTS detection_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id VARCHAR(50) NOT NULL,
    student_id INTEGER,
    register_number VARCHAR(50),
    detection_time TIMESTAMP NOT NULL,
    confidence_score REAL,
    face_location TEXT,  -- JSON: {x, y, w, h}
    embedding_similarity REAL,
    is_recognized BOOLEAN DEFAULT 0,
    frame_number INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE SET NULL
);

-- Fusion events table: Aggregated multi-camera detections
CREATE TABLE IF NOT EXISTS fusion_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    register_number VARCHAR(50) NOT NULL,
    event_time TIMESTAMP NOT NULL,
    total_detections INTEGER,
    camera_count INTEGER,
    average_confidence REAL,
    max_confidence REAL,
    cameras_detected TEXT,  -- JSON array of camera IDs
    decision VARCHAR(50),  -- 'mark_attendance', 'ignore', 'uncertain'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

-- Confidence tracking table: Track confidence over time for decisions
CREATE TABLE IF NOT EXISTS confidence_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    register_number VARCHAR(50) NOT NULL,
    camera_id VARCHAR(50) NOT NULL,
    confidence_score REAL,
    detection_time TIMESTAMP NOT NULL,
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    sample_count INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);

-- System logs table: System events and errors
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    log_level VARCHAR(20),
    component VARCHAR(100),
    message TEXT,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Camera status table: Track camera health and status
CREATE TABLE IF NOT EXISTS camera_status (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id VARCHAR(50) UNIQUE NOT NULL,
    location VARCHAR(255),
    status VARCHAR(20) DEFAULT 'offline',  -- online, offline, error
    last_frame_time TIMESTAMP,
    total_frames_processed INTEGER DEFAULT 0,
    total_detections INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    uptime_seconds INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_students_register ON students(register_number);
CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance_records(student_id, attendance_date);
CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_records(attendance_date);
CREATE INDEX IF NOT EXISTS idx_detection_time ON detection_events(detection_time);
CREATE INDEX IF NOT EXISTS idx_detection_camera ON detection_events(camera_id);
CREATE INDEX IF NOT EXISTS idx_fusion_time ON fusion_events(event_time);
CREATE INDEX IF NOT EXISTS idx_confidence_student ON confidence_tracking(student_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp);

-- Views for analytics

-- Daily attendance summary
CREATE VIEW IF NOT EXISTS daily_attendance_summary AS
SELECT 
    attendance_date,
    COUNT(DISTINCT student_id) as total_present,
    AVG(confidence_score) as avg_confidence,
    COUNT(CASE WHEN status = 'late' THEN 1 END) as late_count
FROM attendance_records
GROUP BY attendance_date;

-- Student attendance history
CREATE VIEW IF NOT EXISTS student_attendance_history AS
SELECT 
    s.register_number,
    s.name,
    COUNT(ar.id) as days_present,
    AVG(ar.confidence_score) as avg_confidence,
    MAX(ar.check_in_time) as last_attendance
FROM students s
LEFT JOIN attendance_records ar ON s.id = ar.student_id
WHERE s.is_active = 1
GROUP BY s.id;

-- Camera performance view
CREATE VIEW IF NOT EXISTS camera_performance AS
SELECT 
    cs.camera_id,
    cs.location,
    cs.status,
    cs.total_frames_processed,
    cs.total_detections,
    CAST(cs.total_detections AS REAL) / NULLIF(cs.total_frames_processed, 0) as detection_rate,
    cs.uptime_seconds / 3600.0 as uptime_hours
FROM camera_status cs;
