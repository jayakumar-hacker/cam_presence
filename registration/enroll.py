"""
Student Registration/Enrollment Module
CLI-based interface for registering new students
Captures multiple face samples and stores embeddings
"""

import cv2
import time
import numpy as np
from typing import Optional, List
from pathlib import Path

import config
from database.db import get_db
from detection.face_detector import FaceDetector
from recognition.embedding import FaceEmbedder


class StudentEnrollment:
    """
    Student enrollment system
    Captures face samples and registers student in database
    """
    
    def __init__(self):
        self.db = get_db()
        self.detector = FaceDetector()
        self.embedder = FaceEmbedder()
        
        print("[Enrollment] System initialized")
    
    def enroll_student_cli(self):
        """CLI interface for enrolling a new student"""
        print("\n" + "="*60)
        print("STUDENT ENROLLMENT SYSTEM")
        print("="*60)
        
        # Get student details
        print("\nEnter student details:")
        register_number = input("Register Number: ").strip()
        name = input("Name: ").strip()
        email = input("Email (optional): ").strip() or None
        department = input("Department (optional): ").strip() or None
        year_str = input("Year (optional): ").strip()
        year = int(year_str) if year_str else None
        
        if not register_number or not name:
            print("[ERROR] Register number and name are required!")
            return False
        
        # Check if student already exists
        existing = self.db.get_student_by_register_number(register_number)
        if existing:
            print(f"\n[WARNING] Student {register_number} already exists!")
            overwrite = input("Overwrite existing data? (y/n): ").strip().lower()
            if overwrite != 'y':
                print("[CANCELLED] Enrollment cancelled")
                return False
        
        # Register student in database
        print(f"\n[INFO] Registering student: {name} ({register_number})")
        student_id = self.db.register_student(
            register_number, name, email, department, year
        )
        
        print(f"[SUCCESS] Student registered with ID: {student_id}")
        
        # Face capture
        print("\n" + "-"*60)
        print("FACE CAPTURE")
        print("-"*60)
        print(f"Required samples: {config.REGISTRATION_SAMPLES_REQUIRED}")
        print("Instructions:")
        print("  - Look directly at camera")
        print("  - Ensure good lighting")
        print("  - Keep face centered")
        print("  - Press 'c' to capture, 'q' to quit")
        print("-"*60)
        
        capture_mode = input("\nCapture from (1) Webcam or (2) Manual images? (1/2): ").strip()
        
        if capture_mode == '1':
            success = self.capture_from_webcam(student_id, register_number, name)
        else:
            success = self.capture_manual(student_id, register_number, name)
        
        if success:
            print(f"\n[SUCCESS] Enrollment complete for {name} ({register_number})")
            print(f"Total embeddings stored: {len(self.db.get_student_embeddings(student_id))}")
            return True
        else:
            print(f"\n[FAILED] Enrollment failed for {name}")
            return False
    
    def capture_from_webcam(self, student_id: int, register_number: str, 
                           name: str) -> bool:
        """Capture face samples from webcam"""
        print("\n[INFO] Starting webcam capture...")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[ERROR] Cannot open webcam")
            return False
        
        samples_collected = 0
        embeddings = []
        
        print("[INFO] Webcam started. Press 'c' to capture, 'q' to quit")
        
        while samples_collected < config.REGISTRATION_SAMPLES_REQUIRED:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Detect faces
            detections = self.detector.detect_faces(frame)
            
            # Draw detections
            display_frame = frame.copy()
            
            for detection in detections:
                x, y, w, h = detection['box']
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                confidence = detection['confidence']
                label = f"Face: {confidence:.2f}"
                cv2.putText(display_frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show progress
            progress = f"Samples: {samples_collected}/{config.REGISTRATION_SAMPLES_REQUIRED}"
            cv2.putText(display_frame, progress, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Student Enrollment - Press C to capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n[CANCELLED] Capture cancelled by user")
                break
            
            if key == ord('c'):
                # Capture sample
                if len(detections) == 0:
                    print("[WARNING] No face detected. Try again.")
                    continue
                
                if len(detections) > 1:
                    print("[WARNING] Multiple faces detected. Ensure only one person.")
                    continue
                
                # Extract face
                detection = detections[0]
                x, y, w, h = detection['box']
                face_roi = self.detector.extract_face_roi(frame, detection['box'])
                
                if face_roi is None:
                    print("[WARNING] Failed to extract face. Try again.")
                    continue
                
                # Generate embedding
                embedding = self.embedder.generate_embedding(face_roi)
                
                if embedding is None:
                    print("[WARNING] Failed to generate embedding. Try again.")
                    continue
                
                # Calculate quality
                quality = self.embedder.calculate_quality(face_roi)
                
                if quality < config.REGISTRATION_QUALITY_THRESHOLD:
                    print(f"[WARNING] Low quality image ({quality:.2f}). Try again.")
                    continue
                
                # Store embedding
                self.db.store_embedding(student_id, embedding, quality)
                embeddings.append(embedding)
                
                samples_collected += 1
                print(f"[SUCCESS] Sample {samples_collected} captured (quality: {quality:.2f})")
        
        cap.release()
        cv2.destroyAllWindows()
        
        if samples_collected >= config.REGISTRATION_SAMPLES_REQUIRED:
            print(f"\n[SUCCESS] Captured {samples_collected} samples")
            return True
        else:
            print(f"\n[FAILED] Only captured {samples_collected} samples")
            return False
    
    def capture_manual(self, student_id: int, register_number: str, 
                      name: str) -> bool:
        """Manual face capture from image files"""
        print("\n[INFO] Manual image capture mode")
        print("You will be prompted to provide paths to face images")
        
        samples_collected = 0
        embeddings = []
        
        while samples_collected < config.REGISTRATION_SAMPLES_REQUIRED:
            print(f"\nSample {samples_collected + 1}/{config.REGISTRATION_SAMPLES_REQUIRED}")
            
            image_path = input("Enter path to face image (or 'q' to quit): ").strip()
            
            if image_path.lower() == 'q':
                print("[CANCELLED] Capture cancelled by user")
                break
            
            # Load image
            if not Path(image_path).exists():
                print("[ERROR] File not found")
                continue
            
            frame = cv2.imread(image_path)
            
            if frame is None:
                print("[ERROR] Failed to load image")
                continue
            
            # Detect faces
            detections = self.detector.detect_faces(frame)
            
            if len(detections) == 0:
                print("[WARNING] No face detected in image")
                continue
            
            if len(detections) > 1:
                print("[WARNING] Multiple faces detected. Use image with single face.")
                continue
            
            # Extract face
            detection = detections[0]
            face_roi = self.detector.extract_face_roi(frame, detection['box'])
            
            if face_roi is None:
                print("[WARNING] Failed to extract face")
                continue
            
            # Generate embedding
            embedding = self.embedder.generate_embedding(face_roi)
            
            if embedding is None:
                print("[WARNING] Failed to generate embedding")
                continue
            
            # Calculate quality
            quality = self.embedder.calculate_quality(face_roi)
            
            if quality < config.REGISTRATION_QUALITY_THRESHOLD:
                print(f"[WARNING] Low quality image ({quality:.2f})")
                retry = input("Use anyway? (y/n): ").strip().lower()
                if retry != 'y':
                    continue
            
            # Store embedding
            self.db.store_embedding(student_id, embedding, quality)
            embeddings.append(embedding)
            
            samples_collected += 1
            print(f"[SUCCESS] Sample {samples_collected} added (quality: {quality:.2f})")
        
        if samples_collected >= config.REGISTRATION_SAMPLES_REQUIRED:
            print(f"\n[SUCCESS] Captured {samples_collected} samples")
            return True
        else:
            print(f"\n[FAILED] Only captured {samples_collected} samples")
            return False
    
    def list_registered_students(self):
        """List all registered students"""
        students = self.db.get_all_students()
        
        if not students:
            print("\n[INFO] No registered students")
            return
        
        print("\n" + "="*60)
        print("REGISTERED STUDENTS")
        print("="*60)
        
        for i, student in enumerate(students, 1):
            embeddings = self.db.get_student_embeddings(student['id'])
            
            print(f"\n{i}. {student['name']}")
            print(f"   Register Number: {student['register_number']}")
            print(f"   Email: {student['email'] or 'N/A'}")
            print(f"   Department: {student['department'] or 'N/A'}")
            print(f"   Year: {student['year'] or 'N/A'}")
            print(f"   Face Samples: {len(embeddings)}")
            print(f"   Enrolled: {student['enrollment_date']}")
        
        print("\n" + "="*60)
        print(f"Total Students: {len(students)}")
        print("="*60)


def run_enrollment_cli():
    """Run the enrollment CLI"""
    enrollment = StudentEnrollment()
    
    while True:
        print("\n" + "="*60)
        print("STUDENT ENROLLMENT MENU")
        print("="*60)
        print("1. Enroll New Student")
        print("2. List Registered Students")
        print("3. Exit")
        print("="*60)
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == '1':
            enrollment.enroll_student_cli()
        elif choice == '2':
            enrollment.list_registered_students()
        elif choice == '3':
            print("\n[INFO] Exiting enrollment system")
            break
        else:
            print("[ERROR] Invalid choice")


if __name__ == "__main__":
    run_enrollment_cli()
