import os
import json
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_pose_from_image(image_bgr):
    """
    Runs Mediapipe Pose on one frame and returns a list of keypoints.
    If no detection: returns None.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False
    ) as pose:

        result = pose.process(image_rgb)

        if not result.pose_landmarks:
            return None

        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })

        return keypoints


def process_frame_folder(frames_folder, output_folder):
    """
    Runs pose estimation on every frame in frames_folder
    and writes pose keypoints to output_folder as JSON.
    """

    os.makedirs(output_folder, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(frames_folder)
        if f.lower().endswith((".jpg", ".png"))
    )

    if not frame_files:
        print(f"No frames found in {frames_folder}")
        return

    print(f"Found {len(frame_files)} frames. Extracting poses...")

    last_valid = None

    for idx, filename in enumerate(frame_files):
        frame_path = os.path.join(frames_folder, filename)
        img = cv2.imread(frame_path)

        if img is None:
            print(f"WARNING: Couldn't read frame {frame_path}")
            continue

        keypoints = extract_pose_from_image(img)

        # Save last valid if current frame can't be detected
        if keypoints is None:
            keypoints = last_valid
        else:
            last_valid = keypoints

        # Save JSON
        out_path = os.path.join(
            output_folder, filename.replace(".jpg", ".json").replace(".png", ".json")
        )

        with open(out_path, "w") as f:
            json.dump({
                "frame": filename,
                "pose_keypoints": keypoints
            }, f)

        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(frame_files)}")

    print("Pose extraction complete.")