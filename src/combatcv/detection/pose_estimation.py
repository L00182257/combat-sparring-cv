import os
import json

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

def extract_pose_from_image(image_bgr):
    #Run MediaPipe Pose on a single BGR image (OpenCV format)
    #and return a list of keypoints, or None if no person is detected.

    #Each keypoint: { 'x': float, 'y': float, 'z': float, 'visibility': float }
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose: #treat each frame independently
        result = pose.process(image_rgb)

        if not result.pose_landmarks:
            return None

        keypoints = []
        for lm in result.pose_landmarks.landmark:
            keypoints.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            })

        return keypoints
    
def process_frame_folder(frames_folder, output_folder):
    #Run pose estimation on every frame in frames_folder
    #and save pose keypoints as JSON files to output_folder.

    #frames_folder: e.g. data/processed/sample_frames
    #output_folder: e.g. data/processed/sample_pose
    
    os.makedirs(output_folder, exist_ok=True)

    frame_files = sorted(
        f for f in os.listdir(frames_folder)
        if f.lower().endswith((".jpg", ".png"))
    )

    if not frame_files:
        print(f"No frame images found in {frames_folder}")
        return

    print(f"Found {len(frame_files)} frames in {frames_folder}")

    for idx, filename in enumerate(frame_files):
        frame_path = os.path.join(frames_folder, filename)
        image = cv2.imread(frame_path)

        if image is None:
            print(f"Warning: could not read image {frame_path}")
            continue

        keypoints = extract_pose_from_image(image)

        json_name = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(output_folder, json_name)

        data = {
            "frame": filename,
            "pose_keypoints": keypoints  # can be None if no pose detected
        }

        with open(json_path, "w") as f:
            json.dump(data, f)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} / {len(frame_files)} frames")

    print(f"Done! Pose JSONs saved to {output_folder}")

if __name__ == "__main__":
    #process the frames just extracted
    frames_folder = "data/processed/sample_frames"
    output_folder = "data/processed/sample_pose"

    process_frame_folder(frames_folder, output_folder)


    

