import cv2
import os

def extract_frames(video_path, output_folder, target_fps=5):
    #Extract frmes from a video at a reduced frame rate
    #Args:
    #  video_path (str): path to the input .mp4 file
    #  output_folder (str): folder where frames will be saved
    #  target_fps (int): how many frames per second to extract
    
    os.makedirs(output_folder, exist_ok=True) #If folder exsists do nothing, if it dosnt make it.

    #Creates vid reader
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    if original_fps == 0:
        original_fps = 30  # safe fallback

    frame_interval = int(original_fps / target_fps)
    frame_count = 0
    saved_count = 0

    while True:
      ret, frame = cap.read()
      if not ret:
        break  # end of video

      if frame_count % frame_interval == 0:
        filename = f"frame_{saved_count:05d}.jpg"
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, frame)
        saved_count += 1

      frame_count += 1

    cap.release()
    print(f"Done! Saved {saved_count} frames to {output_folder}")



     

     


    