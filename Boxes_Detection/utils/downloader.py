import os
import cv2
import numpy as np
from pytubefix import Playlist, YouTube
from tqdm import tqdm

# Correção para extrair URLs corretamente
def get_playlist_urls(url_playlist):
    """
    Gets all the videos urls from a Youtube playlist.
    
    Args:
        get_playlist_urls (str): Youtube playlist url.

    Returns: 
        playlist.urls: Object containning all urls
    """
    playlist = Playlist(url_playlist)
    print(f'Downloading playlist {playlist.title}')
    playlist._video_regex = r"\"url\":\"(/watch\?v=[\w-]*)"
    return playlist.video_urls

def download_videos(urls, pasta_destino="videos"):
    """
    Gets a playlist.urls objects containning a list o videos and download them to a folder.

    Args:
        urls (playlist.urls): List of video urls
        pasta_destino (str): Pestination folder path

    Returns:
        list: The paths for all downloaded videos.
    """
    if not os.path.exists(pasta_destino):
        os.makedirs(pasta_destino)

    paths_videos = []
    for url in tqdm(urls):
        try:
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            video_path = stream.download(output_path=pasta_destino)
            paths_videos.append(video_path)
        except Exception as e:
            print(f"Error when downloading {url}: {e}")

    return paths_videos

def extrair_frames(video_path, n_frames, frames_folder="frames"):
    """
    Get a video_path and save frames as png into the frame_folder.

    Args:
        video_path (str): Path to the video.
        n_frames (int): How many frames do you want.
        frame_folder (str): Path to the folder where frames will the images will be saved
    """
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < n_frames:
        indexes = np.arange(total_frames)
    else:
        indexes = np.linspace(0, total_frames - 1, n_frames, dtype=int)

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for i, frame_id in enumerate(indexes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if ret:
            frame_name = os.path.join(frames_folder, f"{base_name}_frame_{i+1}.jpg")
            cv2.imwrite(frame_name, frame)
        else:
            print(f"Not possible to read frame {frame_id} from video {base_name}")

    cap.release()

def process_playlist(url_playlist, n_frames_por_video):
    """
    Receives a folder with videos files, and execute the extract_frames function for each of them.

    Args:
        url_playlist (str): String containning url to Youtube Playlist
        n_frames_por_video (int): How many frames will be extracted per video.
    """
    urls = get_playlist_urls(url_playlist)
    
    if not urls:
        print("No videos found in the playlist. Check the link.")
        return

    paths_videos = download_videos(urls)

    print("\nExtracting video frames...")
    for caminho in tqdm(paths_videos):
        extrair_frames(caminho, n_frames_por_video)

if __name__ == "__main__":
    URL_PLAYLIST = "https://www.youtube.com/playlist?list=PL3D2CadNPO5-9-2cu97KfDKxfe1OdSy-g"
    N_FRAMES_PER_VIDEO = 5  # ajuste conforme desejado
    yt = YouTube('https://youtube.com/watch?v=2lAe1cqCOXo')
    #print(yt.title)
    process_playlist(URL_PLAYLIST, N_FRAMES_PER_VIDEO)
