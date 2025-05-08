
## Folder structure

```text
Basket_project/
├── Tracker/
│   └── src/
│       └── data/  
│           └── annotations/            # Clip_dict.npy containing all info
│           └── videos/                 # Input/Output game video
│       └── utils/            
│       └── main.py
│
├── Player_Detection/
│   └── src/
│       └── data/  
│           └── annotations/            # Bbox.npy containing all players bounding boxes
│           └── weigths/                # Weights of pre-trained models
│       └── utils/  
│       └── main.py                   # Model to detect players
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

## Installing Requirements

```bash
python3 -m ensurepip --upgrade
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
cd Tracker
git clone https://github.com/meituan/YOLOv6.git
cd YOLOv6
python3 -m pip install -r requirements
cd ../..
python3 -m pip install -r requirements
```

## Running

```bash
./run_pipeline.sh
```
