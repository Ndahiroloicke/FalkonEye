#!/usr/bin/env python3
"""
Example: Activity Logging Demo
Demonstrates how to use the activity logger programmatically.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.activity_logger import ActivityLogger
from src import config


def simulate_activity_session():
    """Simulate a session with various activities."""
    
    print("="*70)
    print("ACTIVITY HISTORY LOGGING DEMO")
    print("="*70)
    
    # Create a logger for a person
    logger = ActivityLogger("DemoPerson", config.HISTORY_DIR)
    
    print("\nSimulating activities...")
    
    # Simulate various activities over time
    activities = [
        (10, "blink", (320, 240)),
        (25, "smile", (322, 242)),
        (40, "blink", (324, 243)),
        (55, "move_right", (360, 242)),  # Moved right
        (70, "smile", (362, 241)),
        (85, "blink", (365, 240)),
        (100, "move_left", (330, 238)),  # Moved left
        (115, "blink", (328, 237)),
        (130, "move_down", (328, 270)),  # Moved down
        (145, "smile", (327, 272)),
        (160, "move_up", (326, 245)),   # Moved up
        (175, "blink", (325, 243)),
    ]
    
    previous_pos = None
    
    for frame_num, activity, position in activities:
        if activity in ["blink", "smile"]:
            # Direct activity logging
            logger.log_activity(activity, frame_num, position)
            print(f"  Frame {frame_num:3d}: {activity.upper():10s} at {position}")
        else:
            # Movement detection (simulated)
            if previous_pos:
                # Manually trigger movement for demo
                logger.log_activity(activity, frame_num, position, 
                                  f"dx={position[0]-previous_pos[0]:.1f}, "
                                  f"dy={position[1]-previous_pos[1]:.1f}")
                logger.activity_counts[activity] += 1
                print(f"  Frame {frame_num:3d}: {activity.upper():10s} to {position}")
        
        previous_pos = position
    
    # Show statistics
    print("\n" + "-"*70)
    stats = logger.get_statistics()
    print(f"\nSession Statistics for {stats['person']}:")
    print(f"  Total Activities: {stats['total']}")
    print(f"\nBreakdown:")
    for activity, count in sorted(stats['counts'].items()):
        if count > 0:
            print(f"  {activity.replace('_', ' ').title()}: {count}")
    
    # Save summary
    print("\n" + "-"*70)
    logger.save_summary()
    
    print("\n" + "="*70)
    print("✓ Demo completed!")
    print(f"\nActivity history saved to: {config.HISTORY_DIR}/")
    print("\nTo view the history logs, run:")
    print("  python src/view_activity_logs.py")
    print("="*70)


if __name__ == "__main__":
    simulate_activity_session()
