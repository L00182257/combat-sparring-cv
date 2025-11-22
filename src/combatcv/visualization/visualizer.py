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

def visualize_video(original_video, pose_folder, punch_frames, output_path):
    cap = cv2.VideoCapture(original_video)
    if not cap.isOpened():
        print("ERROR: Cannot open video.")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build quick punch lookup
    punch_lookup = {frame: side for (side, frame) in punch_frames}

    # Maintain running counts
    left_so_far = 0
    right_so_far = 0

    # Load pose JSON files
    pose_files = sorted(f for f in os.listdir(pose_folder) if f.endswith(".json"))

    if len(pose_files) != frame_count:
        print("Warning: frame count and pose count differ (normal for reduced FPS).")

    current_pose_index = 0

    print("Generating visualization video...")

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # find nearest pose file
        if current_pose_index < len(pose_files) - 1:
            pose_file = pose_files[current_pose_index]
            with open(os.path.join(pose_folder, pose_file)) as f:
                data = json.load(f)
            keypoints = data["pose_keypoints"]

            current_pose_index += 1
        else:
            keypoints = None

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
