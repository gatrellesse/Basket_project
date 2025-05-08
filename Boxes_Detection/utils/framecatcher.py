import os
import cv2
import numpy as np
from tqdm import tqdm

def extract_frames(video_path, n_frames, start_offset=0, end_offset=-1, frame_folder="frames"):
    """
    Get a video_path and save frames as png into the frame_folder.

    Args:
        video_path (str): Path to the video.
        n_frames (int): How many frames do you want.
        start_offset (int): Starting frame.
        end_offset (int): Ending frame.
        frame_folder (str): Path to the folder where frames will the images will be saved
    """
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Converter offsets de segundos para frames
    start_offset_frames = int(offset_start * fps)
    end_offset_frames = int(offset_end * fps)

    start = start_offset_frames
    end = total_frames - end_offset_frames

    if end <= start:
        print(f"Invalid offsets for video {video_path}. Ignoring...")
        cap.release()
        return

    frames_interval = end - start

    if frames_interval < n_frames:
        indexes = np.arange(start, end)
    else:
        indexes = np.linspace(start, end - 1, n_frames, dtype=int)

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i, frame_id in enumerate(indexes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frame_name = os.path.join(frame_folder, f"{base_name}_frame_{i+1}.jpg")
            cv2.imwrite(frame_name, frame)
        else:
            print(f"Not possible to read frame {frame_id} from vídeo {base_name}")

    cap.release()

def process_videos_local(n_frames_por_video, video_folder="videos", start_offset=0, end_offset=0):
    """
    Receives a folder with videos files, and execute the extract_frames function for each of them.

    Args:
        n_frames_por_video (int): How many frames will be extracted per video.
        video_folder (str): Path to the folder containing the video files.
        start_offset (int): Starting frame.
        end_offset (int): Ending frame.
    """
    if not os.path.exists(video_folder):
        print(f"Videos folder '{video_folder}' não encontrada.")
        return

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.webm'))]
    if not video_files:
        print(f"No videos found in folder '{video_folder}'.")
        return

    print("\nExtracting frames from local videos...")
    for file in tqdm(video_files):
        video_path = os.path.join(video_folder, file)
        extract_frames(video_path, n_frames_por_video, start_offset, end_offset)

if __name__ == "__main__":
    N_FRAMES_PER_VIDEO = 500         # número de frames a extrair por vídeo
    OFFSET_START = 60*30              # segundos a ignorar no início
    OFFSET_END = 60*30              # segundos a ignorar no end
    process_videos_local(N_FRAMES_PER_VIDEO, start_offset=OFFSET_STAR, end_offset=OFFSET_END)

