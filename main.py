"""
Smart CCTV Attendance System - Main Entry Point
CLI-based multi-camera face recognition attendance system
"""

import sys
import time
import signal
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from cams.cam_manager import CameraManager
from registration.enroll import run_enrollment_cli
from database.db import get_db


class AttendanceSystem:
    """
    Main attendance system controller
    Manages system lifecycle and CLI interface
    """
    
    def __init__(self):
        self.camera_manager = None
        self.is_running = False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals"""
        print("\n\n[System] Interrupt received, shutting down...")
        self.stop()
        sys.exit(0)
    
    def initialize(self):
        """Initialize the system"""
        print("\n" + "=" * 60)
        print(f"{config.SYSTEM_NAME} v{config.VERSION}")
        print("=" * 60)
        print("Initializing system components...")
        print()
        
        # Initialize database
        print("✓ Database initialized")
        
        # Create camera manager
        self.camera_manager = CameraManager()
        
        # Initialize cameras
        self.camera_manager.initialize_cameras()
        
        print("\n✓ System initialization complete")
    
    def start(self):
        """Start the attendance system"""
        if self.is_running:
            print("[System] Already running")
            return
        
        print("\n[System] Starting attendance monitoring...")
        
        self.camera_manager.start_all()
        self.is_running = True
        
        # Monitor loop
        self._monitor_loop()
    
    def stop(self):
        """Stop the attendance system"""
        if not self.is_running:
            return
        
        print("\n[System] Stopping attendance system...")
        
        self.is_running = False
        
        if self.camera_manager:
            self.camera_manager.stop_all()
        
        print("[System] System stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        last_stats_time = time.time()
        stats_interval = 30  # Print stats every 30 seconds
        
        try:
            while self.is_running:
                time.sleep(1)
                
                # Print statistics periodically
                if time.time() - last_stats_time >= stats_interval:
                    self.camera_manager.print_statistics()
                    last_stats_time = time.time()
        
        except KeyboardInterrupt:
            pass
    
    def show_menu(self):
        """Show main menu"""
        while True:
            print("\n" + "=" * 60)
            print("SMART CCTV ATTENDANCE SYSTEM - MAIN MENU")
            print("=" * 60)
            print("1. Start Attendance Monitoring")
            print("2. Register New Student")
            print("3. View Today's Attendance")
            print("4. View System Statistics")
            print("5. Health Check")
            print("6. Exit")
            print("=" * 60)
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                self.start()
            elif choice == '2':
                self.register_student()
            elif choice == '3':
                self.view_attendance()
            elif choice == '4':
                self.view_statistics()
            elif choice == '5':
                self.health_check()
            elif choice == '6':
                print("\n[System] Exiting...")
                if self.is_running:
                    self.stop()
                break
            else:
                print("[ERROR] Invalid choice")
    
    def register_student(self):
        """Student registration interface"""
        print("\n" + "=" * 60)
        print("STUDENT REGISTRATION")
        print("=" * 60)
        
        from registration.enroll import StudentEnrollment
        enrollment = StudentEnrollment()
        enrollment.enroll_student_cli()
    
    def view_attendance(self):
        """View today's attendance"""
        if self.camera_manager:
            self.camera_manager.print_today_attendance()
        else:
            # Direct database query
            db = get_db()
            records = db.get_today_attendance()
            
            print("\n" + "=" * 60)
            print("TODAY'S ATTENDANCE")
            print("=" * 60)
            print(f"Total Present: {len(records)}")
            print()
            
            if records:
                print(f"{'No.':<5} {'Register':<15} {'Name':<25} {'Time':<10}")
                print("-" * 60)
                
                for i, record in enumerate(records, 1):
                    check_in = record['check_in_time']
                    if isinstance(check_in, str):
                        time_str = check_in.split()[1][:8]
                    else:
                        time_str = check_in.strftime('%H:%M:%S')
                    
                    print(f"{i:<5} {record['register_number']:<15} "
                          f"{record['name']:<25} {time_str:<10}")
            else:
                print("No attendance marked yet today")
            
            print("=" * 60)
    
    def view_statistics(self):
        """View system statistics"""
        if self.camera_manager and self.is_running:
            self.camera_manager.print_statistics()
        else:
            db = get_db()
            stats = db.get_statistics()
            
            print("\n" + "=" * 60)
            print("SYSTEM STATISTICS")
            print("=" * 60)
            print(f"Total Students: {stats['total_students']}")
            print(f"Total Face Embeddings: {stats['total_embeddings']}")
            print(f"Today's Attendance: {stats['today_attendance']}")
            print(f"Total Detection Events: {stats['total_detections']}")
            print("=" * 60)
    
    def health_check(self):
        """Perform system health check"""
        if self.camera_manager:
            self.camera_manager.health_check()
        else:
            print("\n[System] System not running. Start monitoring first.")


def main():
    """Main entry point"""
    system = AttendanceSystem()
    
    try:
        # Initialize system
        system.initialize()
        
        # Show menu
        system.show_menu()
    
    except Exception as e:
        print(f"\n[ERROR] System error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        if system.is_running:
            system.stop()
        
        print("\n[System] Goodbye!")


if __name__ == "__main__":
    main()
