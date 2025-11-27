import cv2
import json
import os

#Draw pose skeleton
# MediaPipe pose landmark connections
BODY_CONNECTIONS = [
    (11, 13), (13, 15),   # Right arm
    (12, 14), (14, 16),   # Left arm
    (11, 12),             # Shoulders
]

def draw_pose(image, keypoints, punch_left=False, punch_right=False):
    h, w, _ = image.shape

    # Draw main skeleton lines
    for a, b in BODY_CONNECTIONS:
        if keypoints[a] and keypoints[b]:
            x1, y1 = int(keypoints[a]["x"] * w), int(keypoints[a]["y"] * h)
            x2, y2 = int(keypoints[b]["x"] * w), int(keypoints[b]["y"] * h)
            cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Draw wrists
    # right wrist = 16
    # left wrist = 15

    rw = keypoints[16]
    lw = keypoints[15]

    if rw:
        rx, ry = int(rw["x"] * w), int(rw["y"] * h)
        color = (0, 0, 255) if punch_right else (0, 255, 255)
        cv2.circle(image, (rx, ry), 10, color, -1)

    if lw:
        lx, ly = int(lw["x"] * w), int(lw["y"] * h)
        color = (255, 0, 0) if punch_left else (255, 255, 0)
        cv2.circle(image, (lx, ly), 10, color, -1)

def visualize_video(original_video, pose_folder, punch_frames, output_path, pose_fps=5):
    """
    original_video: path to the original mp4
    pose_folder: folder with pose JSONs (saved at pose_fps)
    punch_frames: list of (side, pose_frame_index) from motion_analysis
    pose_fps: fps used when extracting frames / poses (e.g. 5)
    """

    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps == 0:
      fps = 30  # fallback

    frame_interval = max(1, int(round(fps / pose_fps)))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build quick punch lookup
    # Map pose-frame punches onto video-frame indices
    punch_lookup = {}
    for side, pose_frame_idx in punch_frames:
      video_frame_idx = pose_frame_idx * frame_interval
      punch_lookup[video_frame_idx] = side

    # Maintain running counts
    left_so_far = 0
    right_so_far = 0

    # Load pose JSON files
    pose_files = sorted(f for f in os.listdir(pose_folder) if f.endswith(".json"))

    if len(pose_files) != frame_count:
        print("Warning: frame count and pose count differ (normal for reduced FPS).")

    num_pose_frames = len(pose_files)

    print("Generating visualization video...")

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Decide which pose frame to use for this video frame
        pose_index = i // frame_interval
        if pose_index >= num_pose_frames:
            keypoints = None
        else:
            pose_file = pose_files[pose_index]
            with open(os.path.join(pose_folder, pose_file)) as f:
                data = json.load(f)
            keypoints = data["pose_keypoints"]

        punch_left = False
        punch_right = False

        if i in punch_lookup:
            side = punch_lookup[i]
            if side == "left":
                left_so_far += 1
                punch_left = True
            else:
                right_so_far += 1
                punch_right = True

        # Draw pose overlay
        if keypoints:
            draw_pose(frame, keypoints, punch_left, punch_right)

        # Add text overlay
        cv2.putText(frame, f"Frame: {i}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.putText(frame, f"Left: {left_so_far}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)

        cv2.putText(frame, f"Right: {right_so_far}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 150), 2)

        cv2.putText(frame, f"Total: {left_so_far + right_so_far}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 0), 2)    

        out.write(frame)

    cap.release()
    out.release()
    print(f"Visualization saved: {output_path}")
