import argparse
import os
import json
import uuid

from combatcv.preprocessing.extract_frames import extract_frames
from combatcv.detection.pose_estimation import process_frame_folder
from combatcv.visualization.visualizer import visualize_video
from combatcv.detection.motion_analysis import (
    load_pose_sequence,
    compute_wrist_speeds,
    detect_punches,
    merge_close_punches
)

#Generate unique output folders
#Each run gets its own folder
def make_output_folders(base="data/processed"):
    run_id = str(uuid.uuid4())[:8]  # short unique id
    frame_dir = os.path.join(base, f"frames_{run_id}")
    pose_dir = os.path.join(base, f"pose_{run_id}")
    result_dir = os.path.join("data", "results")
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    return frame_dir, pose_dir, result_dir

def process_video(video_path, target_fps=5):
    print(f"Processing video: {video_path}")

    frame_dir, pose_dir, result_dir = make_output_folders()

    # 1. Extract frames
    print("Extracting frames...")
    extract_frames(video_path, frame_dir, target_fps=target_fps)

    # 2. Extract pose for each frame
    print("Extracting pose...")
    process_frame_folder(frame_dir, pose_dir)

    # 3. Load pose sequence
    print("Loading pose sequence...")
    sequence = load_pose_sequence(pose_dir)

    # 4. Compute wrist speeds
    print("Computing wrist speeds...")
    speeds = compute_wrist_speeds(sequence)

    # 5. Detect punches
    print("Detecting punches...")
    raw = detect_punches(speeds)
    punches = merge_close_punches(raw)

    # 6. Count left/right
    left_count = sum(1 for side, _ in punches if side == "left")
    right_count = sum(1 for side, _ in punches if side == "right")

    results = {
        "video": video_path,
        "total_punches": len(punches),
        "left_punches": left_count,
        "right_punches": right_count,
        "punch_frames": punches,
        "frame_folder": frame_dir,
        "pose_folder": pose_dir
    }

    # 7. Save results
    filename = os.path.basename(video_path)
    base = os.path.splitext(filename)[0]
    output_file = os.path.join(result_dir, f"{base}_results.json")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    print(f"Total punches: {len(punches)}")

        # 8. Create visualization output path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    viz_output = os.path.join(result_dir, f"{video_name}_visualized.mp4")

    # 9. Generate visualization video
    visualize_video(video_path, pose_dir, punches, viz_output)

    print(f"Visualization saved to {viz_output}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Punch Counter")
    parser.add_argument("--video", type=str, required=True,
                        help="Path to the input video file")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frame extraction FPS")
    args = parser.parse_args()

    process_video(args.video, target_fps=args.fps)
