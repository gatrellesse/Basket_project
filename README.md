
## Folder structure

```text
Basket_project/
├── TacTic/
│   └── src/
│       └── data/  
│           └── annotations/            # Clip_dict.npy containing all info
│           └── videos/                 # Input game video
│       └── utils/  
│           └── pitch_utils.py              
│           └── track_utils.py              
│       └── main.py
│       └── team.py
│       └── render_track.py
│
├── Player_Detection/
│   └── src/
│       └── data/  
│           └── annotations/            # Bbox.npy containing all players bounding boxes
│           └── weigths/                # Weights of pre-trained models
│       └── utils/  
│           └── yolov6_utils.py               
│       └── playes.py                   # Model to detect players
│
├── Terrain_Detection/
│   └── src/
│       └── data/  
│           └── annotations/            # Pitch manually annotated
│           └── input_imgs/             # Imgs used as reference
│           └── output/                 # Imgs output from scripts
│           └── videos/                 # Output videos
│       └── old_prediction/             # Old key points detectors used
│       └── pre_processing/             # Annotation scripts
│       └── pos_processing/             # Annotation points fixer
│       └── prediction/                 # Terrain prediction
│
```


Variable paths to fix:

- players.py: PLAYER_DETECTION_MODEL_PATH, video_in, lib input, local_checkpoint
- all imports

## Pipeline:


- **Terrain Detection**
  - **File**: `Hs.npy`
  - **Description**: Terrain hemography stored in the `Hs.npy` file.
  - Runned by makefile
  
- **Player Detection**
  - **File**: `players.py`
  - **Function**: `func_box()` --> change for a best.py detector
  - **Output**: `bbox.npy` (bounding box coordinates)

- **TacTic** 
  - **Track Follower**
    - **File**: `track_utils.py`
    - **Function**: `run_sv_tracker(bbox.npy)` followed by `box_and_track(bboxes, tracks)`
    - **Output**: `clip_dict.npy` =  `bbox` and `tracks_ids`
    
  - **On-Pitch Player Detection**
    - **File**: `pitch_utils.py`
    - **Function**: `on_pitch(clip_dict[bbox].npy, Hs.npy)`
    - **Output**: `clip_dict.npy` + player positions (`xy`) = `bboxes`, `tracks_ids` and `xy`

  - **Track in pitch**
    - **File**: `track_utils.py`
    - **Function**: `track_in_pitch(clip_dict)`
    - **Output**:  `clip_dict.npy` +  player inside the pitch(`in_pitch`) == `bboxes`, `tracks_ids`, `xy` and `in_pitch`

- **main** 
  - Run if the 5 flags = True 

