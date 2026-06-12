"""
SRS evidence logging: speaker ID, confidence, timestamps, motor commands.

Writes assessor-ready CSV + JSON session summaries under data/history/.
"""

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .srs_commands import TrackingCommand, motor_command_label


class EvidenceLogger:
    """Unified session log for recognition + tracking + motor control."""

    FIELDNAMES = [
        "timestamp",
        "frame_number",
        "speaker_id",
        "confidence",
        "distance",
        "system_state",
        "tracking_command",
        "motor_command",
        "servo_angle",
        "servo_target",
        "face_center_x",
        "face_center_y",
        "frame_width",
        "mqtt_connected",
        "details",
    ]

    def __init__(self, speaker_id: str, output_dir: Path = None):
        self.speaker_id = speaker_id
        self.output_dir = output_dir or config.HISTORY_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{speaker_id}_{stamp}"
        self.csv_path = self.output_dir / f"{self.session_id}_evidence.csv"
        self.json_path = self.output_dir / f"{self.session_id}_evidence_summary.json"

        self._csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

        self._start_time = time.time()
        self._event_counts: Dict[str, int] = {}
        self._last_command: Optional[str] = None
        self._rows = 0

    @staticmethod
    def _iso_timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")

    def log_frame(
        self,
        *,
        frame_number: int,
        confidence: float,
        distance: float,
        system_state: str,
        tracking_command: TrackingCommand,
        servo_angle: float,
        servo_target: Optional[int] = None,
        face_center_x: Optional[float] = None,
        face_center_y: Optional[float] = None,
        frame_width: int = 0,
        mqtt_connected: bool = False,
        details: str = "",
        force: bool = False,
    ) -> None:
        """Log a row; by default only on command/state changes to keep files small."""
        cmd = tracking_command.value
        motor = motor_command_label(tracking_command)
        changed = cmd != self._last_command or force

        if not changed and system_state not in ("SEARCHING", "LOCKED"):
            return

        self._last_command = cmd
        self._event_counts[cmd] = self._event_counts.get(cmd, 0) + 1
        self._rows += 1

        row = {
            "timestamp": self._iso_timestamp(),
            "frame_number": frame_number,
            "speaker_id": self.speaker_id,
            "confidence": f"{confidence:.4f}",
            "distance": f"{distance:.4f}",
            "system_state": system_state,
            "tracking_command": cmd,
            "motor_command": motor,
            "servo_angle": int(round(servo_angle)),
            "servo_target": "" if servo_target is None else int(servo_target),
            "face_center_x": "" if face_center_x is None else f"{face_center_x:.1f}",
            "face_center_y": "" if face_center_y is None else f"{face_center_y:.1f}",
            "frame_width": frame_width,
            "mqtt_connected": mqtt_connected,
            "details": details,
        }
        self._writer.writerow(row)
        self._csv_file.flush()

    def save_summary(self) -> Dict[str, Any]:
        duration = time.time() - self._start_time
        summary = {
            "session_id": self.session_id,
            "speaker_id": self.speaker_id,
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(),
            "duration_seconds": round(duration, 2),
            "total_logged_events": self._rows,
            "tracking_command_counts": self._event_counts,
            "evidence_csv": str(self.csv_path),
            "srs_compliance": {
                "speaker_id_logged": True,
                "confidence_logged": True,
                "timestamps_logged": True,
                "motor_commands_logged": True,
            },
        }
        self.json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def close(self) -> Dict[str, Any]:
        summary = self.save_summary()
        self._csv_file.close()
        return summary
