import json
import os

from combatcv.detection.motion_analysis import (
    load_pose_sequence,
    compute_wrist_speeds,
    detect_punches,
    merge_close_punches
)

def count_punches(pose_folder, output_file=None):

    # 1. Load pose sequence
    sequence = load_pose_sequence(pose_folder)

    # 2. Compute wrist speeds
    speeds = compute_wrist_speeds(sequence)

    # 3. Raw punch detections
    raw_punches = detect_punches(speeds)

    # 4. Merge close punches to avoid double-counting
    punches = merge_close_punches(raw_punches)

    # 5. Count left & right punches
    left_count = sum(1 for side, _ in punches if side == "left")
    right_count = sum(1 for side, _ in punches if side == "right")
    total = len(punches)

    # 6. Prepare result dictionary
    results = {
        "total_punches": total,
        "left_punches": left_count,
        "right_punches": right_count,
        "punch_frames": punches
    }

    # 7. Save to file if needed
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {output_file}")

    return results

if __name__ == "__main__":
    pose_folder = "data/processed/sample_pose"
    output_file = "data/processed/sample_results.json"

    results = count_punches(pose_folder, output_file)

    print("---- Punch Results ----")
    print(f"Total punches: {results['total_punches']}")
    print(f"Left punches: {results['left_punches']}")
    print(f"Right punches: {results['right_punches']}")