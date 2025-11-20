import os
import json
import math

def load_pose_sequence(pose_folder):
    files = sorted(f for f in os.listdir(pose_folder) if f.endswith(".json"))
    sequence = []

    last_valid = None

    for f in files:
        path = os.path.join(pose_folder, f)
        with open(path, "r") as fp:
            data = json.load(fp)

        keypoints = data["pose_keypoints"]

        # If None, use last valid frame
        if keypoints is None:
            sequence.append(last_valid)
        else:
            sequence.append(keypoints)
            last_valid = keypoints

    return sequence

def point_distance(p1, p2): #Hand speed = distance between wrist in frame N and frame N+1.
    return math.sqrt(
        (p1["x"] - p2["x"])**2 +
        (p1["y"] - p2["y"])**2 +
        (p1["z"] - p2["z"])**2
    )

def compute_wrist_speeds(sequence):
    speeds = []

    for i in range(len(sequence) - 1):
        frame1 = sequence[i]
        frame2 = sequence[i+1]

        # Safety check
        if frame1 is None or frame2 is None:
            speeds.append((0, 0))
            continue

        # wrist IDs (MediaPipe):
        # right_wrist = 16
        # left_wrist  = 15
        rw1 = frame1[16]
        rw2 = frame2[16]

        lw1 = frame1[15]
        lw2 = frame2[15]

        right_speed = point_distance(rw1, rw2)
        left_speed  = point_distance(lw1, lw2)

        speeds.append((right_speed, left_speed))

    return speeds

#threshold of 0.02 works well for MediaPipe
def detect_punches(wrist_speeds, threshold=0.02): #Punches = when wrist moves very fast forward.
    punches = []
    for i, (r_speed, l_speed) in enumerate(wrist_speeds):

        punch_frame = None
        
        if r_speed > threshold:
            punch_frame = ("right", i)
        
        if l_speed > threshold:
            punch_frame = ("left", i)
        
        if punch_frame:
            punches.append(punch_frame)

    return punches

#Punches may create multiple fast frames → we want one punch per burst.
def merge_close_punches(punch_list, min_gap=5):
    if not punch_list:
        return []

    merged = [punch_list[0]]

    for side, frame in punch_list[1:]:
        last_side, last_frame = merged[-1]
        if frame - last_frame > min_gap:
            merged.append((side, frame))

    return merged

if __name__ == "__main__":
    pose_folder = "data/processed/sample_pose"

    sequence = load_pose_sequence(pose_folder)
    speeds = compute_wrist_speeds(sequence)
    raw_punches = detect_punches(speeds)
    clean_punches = merge_close_punches(raw_punches)

    print(f"Raw punches detected: {len(raw_punches)}")
    print(f"Clean punch count: {len(clean_punches)}")
    print(clean_punches)
