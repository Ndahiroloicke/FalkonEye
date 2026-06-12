"""
Activity Logger: Track and save activities of the locked person.
Records blinks, smiles, and face movements (left/right, up/down).
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np


class ActivityLogger:
    """Log activities of the locked person."""
    
    def __init__(self, person_name: str, log_dir: Path):
        """
        Initialize activity logger for a specific person.
        
        Args:
            person_name: Name of the locked person
            log_dir: Directory to save activity logs
        """
        self.person_name = person_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = self.log_dir / f"{person_name}_{timestamp}_activities.csv"
        self.json_path = self.log_dir / f"{person_name}_{timestamp}_summary.json"
        
        # Initialize CSV file
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame_number", "activity_type", 
                "face_center_x", "face_center_y", "details"
            ])
        
        # Activity counters
        self.activity_counts = {
            "blink": 0,
            "smile": 0,
            "move_left": 0,
            "move_right": 0,
            "move_up": 0,
            "move_down": 0,
        }
        
        # Movement tracking
        self.previous_face_center: Optional[Tuple[float, float]] = None
        self.movement_threshold_x = 20  # pixels (INCREASED for less jitter)
        self.movement_threshold_y = 20  # pixels (INCREASED for less jitter)
        self.movement_cooldown = 8  # frames between movement detections (INCREASED to reduce noise)
        self.last_movement_frame: Dict[str, int] = {}
        
        # Session start time
        self.session_start = datetime.now()
        
        print(f"✓ Activity history logging started for {person_name}")
        print(f"  CSV log: {self.csv_path}")
        print(f"  Summary: {self.json_path}")
    
    def log_activity(
        self, 
        activity_type: str, 
        frame_number: int,
        face_center: Optional[Tuple[float, float]] = None,
        details: str = ""
    ):
        """
        Log an activity to CSV file.
        
        Args:
            activity_type: Type of activity (blink, smile, move_left, etc.)
            frame_number: Current frame number
            face_center: (x, y) center of face, if available
            details: Additional details about the activity
        """
        timestamp = datetime.now().isoformat()
        
        face_x = face_center[0] if face_center else ""
        face_y = face_center[1] if face_center else ""
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, frame_number, activity_type,
                face_x, face_y, details
            ])
        
        # Update counters
        if activity_type in self.activity_counts:
            self.activity_counts[activity_type] += 1
    
    def detect_and_log_movement(
        self,
        face_center: Tuple[float, float],
        frame_number: int
    ) -> List[str]:
        """
        Detect face movement (left/right/up/down) and log it.
        
        Args:
            face_center: (x, y) center of face
            frame_number: Current frame number
            
        Returns:
            List of detected movements this frame
        """
        movements = []
        
        if self.previous_face_center is None:
            self.previous_face_center = face_center
            return movements
        
        prev_x, prev_y = self.previous_face_center
        curr_x, curr_y = face_center
        
        dx = curr_x - prev_x
        dy = curr_y - prev_y
        
        # Detect horizontal movement
        if abs(dx) > self.movement_threshold_x:
            if dx > 0:  # Moved right
                if frame_number - self.last_movement_frame.get("move_right", -999) >= self.movement_cooldown:
                    movements.append("move_right")
                    self.log_activity(
                        "move_right", frame_number, face_center,
                        f"dx={dx:.1f}px"
                    )
                    self.last_movement_frame["move_right"] = frame_number
                    print(f"➡️  DETECTED: Move RIGHT (dx={dx:.1f}px)")
            else:  # Moved left
                if frame_number - self.last_movement_frame.get("move_left", -999) >= self.movement_cooldown:
                    movements.append("move_left")
                    self.log_activity(
                        "move_left", frame_number, face_center,
                        f"dx={dx:.1f}px"
                    )
                    self.last_movement_frame["move_left"] = frame_number
                    print(f"⬅️  DETECTED: Move LEFT (dx={dx:.1f}px)")
        
        # Detect vertical movement
        if abs(dy) > self.movement_threshold_y:
            if dy > 0:  # Moved down
                if frame_number - self.last_movement_frame.get("move_down", -999) >= self.movement_cooldown:
                    movements.append("move_down")
                    self.log_activity(
                        "move_down", frame_number, face_center,
                        f"dy={dy:.1f}px"
                    )
                    self.last_movement_frame["move_down"] = frame_number
                    print(f"⬇️  DETECTED: Move DOWN (dy={dy:.1f}px)")
            else:  # Moved up
                if frame_number - self.last_movement_frame.get("move_up", -999) >= self.movement_cooldown:
                    movements.append("move_up")
                    self.log_activity(
                        "move_up", frame_number, face_center,
                        f"dy={dy:.1f}px"
                    )
                    self.last_movement_frame["move_up"] = frame_number
                    print(f"⬆️  DETECTED: Move UP (dy={dy:.1f}px)")
        
        # Update previous position
        self.previous_face_center = face_center
        
        return movements
    
    def save_summary(self):
        """Save activity summary to JSON file."""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        
        summary = {
            "person_name": self.person_name,
            "session_start": self.session_start.isoformat(),
            "session_end": datetime.now().isoformat(),
            "session_duration_seconds": round(session_duration, 2),
            "activity_counts": self.activity_counts.copy(),
            "total_activities": sum(self.activity_counts.values()),
            "csv_log": str(self.csv_path),
        }
        
        with open(self.json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Activity summary saved:")
        print(f"  Person: {self.person_name}")
        print(f"  Duration: {session_duration:.1f}s")
        print(f"  Activities:")
        for activity, count in self.activity_counts.items():
            if count > 0:
                print(f"    - {activity}: {count}")
        print(f"  Summary file: {self.json_path}")
    
    def get_statistics(self) -> Dict:
        """Get current activity statistics."""
        return {
            "person": self.person_name,
            "counts": self.activity_counts.copy(),
            "total": sum(self.activity_counts.values()),
        }
