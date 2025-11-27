import os
import json
import math

VIS_THRESHOLD = 0.6

def smooth_speeds(speeds, window=5):
    smoothed = []
    for i in range(len(speeds)):
        start = max(0, i - window)
        window_vals = speeds[start:i+1]

        r_avg = sum(v[0] for v in window_vals) / len(window_vals)
        l_avg = sum(v[1] for v in window_vals) / len(window_vals)

        smoothed.append((r_avg, l_avg))
    return smoothed

def load_pose_sequence(pose_folder):
    files = sorted(f for f in os.listdir(pose_folder) if f.endswith(".json"))
    sequence = []
    last_valid = None

    for file in files:
        path = os.path.join(pose_folder, file)
        with open(path, "r") as f:
            data = json.load(f)

        kp = data["pose_keypoints"]
        if kp is None:
            kp = last_valid
        else:
            last_valid = kp

        sequence.append(kp)

    return sequence

def point_distance(a, b):
    return math.sqrt(
        (a["x"] - b["x"])**2 +
        (a["y"] - b["y"])**2 +
        (a["z"] - b["z"])**2
    )

def compute_wrist_speeds(sequence):
    speeds = []

    for i in range(len(sequence) - 1):

        frame1 = sequence[i]
        frame2 = sequence[i + 1]

        if frame1 is None or frame2 is None:
            speeds.append((0.0, 0.0))
            continue

        rw1 = frame1[16]   # right wrist
        rw2 = frame2[16]
        lw1 = frame1[15]   # left wrist
        lw2 = frame2[15]

        if (rw1 is None or rw2 is None or lw1 is None or lw2 is None):
            speeds.append((0.0, 0.0))
            continue

        rs = point_distance(rw1, rw2)
        ls = point_distance(lw1, lw2)

        speeds.append((float(rs), float(ls)))

    return speeds

def forward_speed(a, b):
    dz = a["z"] - b["z"]
    return max(0, -dz)  # moving toward camera

def detect_punches(smoothed_speeds, sequence,
                   speed_thresh=0.02,
                   forward_thresh=0.003,
                   cooldown=5):

    punches = []
    last_punch = -99

    for i in range(len(smoothed_speeds)):

        frame = sequence[i]
        if frame is None:
            continue

        r_speed, l_speed = smoothed_speeds[i]
        rw = frame[16]
        lw = frame[15]

        if rw["visibility"] < VIS_THRESHOLD or lw["visibility"] < VIS_THRESHOLD:
            continue

        if i < len(sequence) - 1:
            fwd_r = forward_speed(rw, sequence[i+1][16])
            fwd_l = forward_speed(lw, sequence[i+1][15])
        else:
            fwd_r = fwd_l = 0

        punch = None

        if r_speed > speed_thresh and fwd_r > forward_thresh:
            punch = ("right", i)
        if l_speed > speed_thresh and fwd_l > forward_thresh:
            punch = ("left", i)

        if punch and (i - last_punch > cooldown):
            punches.append(punch)
            last_punch = i

    return punches

def merge_close_punches(punch_list, min_gap=5):
    if not punch_list:
        return []

    merged = [punch_list[0]]

    for side, idx in punch_list[1:]:
        last_side, last_idx = merged[-1]
        if idx - last_idx > min_gap:
            merged.append((side, idx))

    return merged