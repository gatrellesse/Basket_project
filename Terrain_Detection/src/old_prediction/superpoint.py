import os
import torch
import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from transformers import AutoImageProcessor, SuperPointForKeypointDetection

# if out of memory
"""
import gc
del model
gc.collect()
torch.cuda.empty_cache()
"""

MIN_MATCH_COUNT = 10
i_frame = [104700, 104700+75, 104700+75+35]

current_dir = Path(__file__).parent.resolve()
base_path = current_dir.parent 
base_path = base_path / "data"
images_folder = base_path / "input_imgs"
videos_folder = base_path / "videos"
annotations_folder = base_path / "annotations"

imgs = []
for i in i_frame:
    img_path = images_folder / f"img_{i}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    imgs.append(img)

size_ratio = 15
conf_thresh = 10
plot_pts = False


video_in = videos_folder / "basket_game.mp4"
Hs_name = annotations_folder / f"Hs_supt{size_ratio}.npy"
video_out = videos_folder / f"pitch_supt{size_ratio}.mp4"


#pitch = np.load('pitch.npy')
conf_thresh /= 100
if size_ratio > 10: size_ratio /= 10


def calc_ref_hist(imgs):
    return np.hstack([cv2.calcHist([img], [0], None, [256], [0, 256]) for img in imgs])

def best_match(new_img, ref_hist):
     new_hist = cv2.calcHist([new_img], [0], None, [256], [0, 256])

     match_probs = [cv2.matchTemplate(hist, new_hist, cv2.TM_CCOEFF_NORMED)[0][0] for hist in ref_hist.T]
     match_probs = np.array(match_probs)
     return np.argmax(match_probs), match_probs

def load_data(img_name, pts_name):
    """Load image and points data."""
    data = np.load(pts_name, allow_pickle=True).item()
    img = cv2.imread(img_name)
    print(img_name)
    pts = data["pts"]
    idents = data["ident"]
    return img, pts, idents




device = "cuda:0" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(
    "magic-leap-community/superpoint", 
    use_fast=True
)
# image_processing_superpoint.py 
# dans /anaconda3/lib/python3.12/site-package/transformers/models/superpoints
# ligne 135 self.do_resize = do_resize mise en commantaire pour forcer à False
model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
model = model.to(device)



FLANN_INDEX_KDTREE = 1 # beaucoup plus long si on met 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
imgs=[]
annots = []
annots_idx = []
for i in i_frame:
    annots_name = annotations_folder / f"pts_dict_{i}.npy"
    imgs_path = images_folder / f"img_{i}.png"
    img = cv2.imread(imgs_path)
    img, pts, idents = load_data(imgs_path, annots_name)
    imgs.append(img)
    annots.append(pts)
    annots_idx.append(idents)



h, w = img.shape[:2]
if size_ratio != 1: w_resize, h_resize = int(w / size_ratio), int(h / size_ratio)
ref_hist = calc_ref_hist(imgs)

if size_ratio == 1: rgbs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
else:
    rgbs=[]
    for img in imgs:
        img_r = cv2.resize(img, (w_resize, h_resize))
        rgbs.append(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
        
with torch.no_grad():
    inputs = processor(rgbs, return_tensors="pt").to(device)
    outputs = model(**inputs)

image_sizes = torch.tile(torch.tensor([h_resize,w_resize]), (len(rgbs),1)).to(device)
outputs = processor.post_process_keypoint_detection(outputs, image_sizes)

kpts_ref = []
descs_ref = []
for output in outputs:
    kp = output['keypoints'].to('cpu').numpy()
    desc = output['descriptors'].to('cpu').numpy()
    scores = output['scores'].to('cpu').numpy() # filter with a threshold ?
    good_scores = scores > conf_thresh
    kp = kp[good_scores]
    desc = desc[good_scores]
    outboard = np.logical_not((kp[:,1] > 875) * (kp[:,0] < 325))
    kpts_ref.append(kp[outboard])
    descs_ref.append(desc[outboard])

init_frame = 100_000
avi_name =  videos_folder/ f"results.avi"
video_capture = cv2.VideoCapture()
if video_capture.open( video_in ):
    print("video carregado")
    w, h = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 
    ret, frame = video_capture.read()
with torch.no_grad():
    batch_size = 4
    t0 = time.time()
    Hs = []
    t_match = 0
    for i in range(0,2_000,batch_size):
        imgs = []
        hist_matches = []
        for i_batch in range(batch_size):
            ret, frame = video_capture.read()
            if not ret: break
            i_match, probs = best_match(frame, ref_hist)
            hist_matches.append(i_match)
            imgs.append(frame)
        rgbs=[]
        if size_ratio == 1: rgbs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs]
        else:
            for img in imgs:
                img_r = cv2.resize(img, (w_resize, h_resize))
                rgbs.append(cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB))
        inputs = processor(rgbs, return_tensors="pt").to(device)
        outputs = model(**inputs)
        image_sizes = torch.tile(torch.tensor([h_resize,w_resize]), (len(rgbs),1)).to(device)
        outputs = processor.post_process_keypoint_detection(outputs, image_sizes)
        
        for i_match, output  in zip(hist_matches, outputs):
            kp = output['keypoints'].to('cpu').numpy()
            desc = output['descriptors'].to('cpu').numpy()
            scores = output['scores'].to('cpu').numpy() # filter with a threshold ?
            good_scores = scores > conf_thresh
            kp = kp[good_scores]
            desc = desc[good_scores]
            outboard = np.logical_not((kp[:,1] > 875) * (kp[:,0] < 325))
            kp = kp[outboard]
            desc = desc[outboard]

            
            t1 = time.time()
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descs_ref[i_match], desc,k=2)
            
            good = []
            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            
            t_match += time.time() - t1
            if len(good)>MIN_MATCH_COUNT:
                src_pts = np.float32([ kpts_ref[i_match][m.queryIdx] for m in good]).reshape(-1,1,2)
                dst_pts = np.float32([ kp[m.trainIdx] for m in good ]).reshape(-1,1,2)
            Mratio, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

            M =  np.diag([size_ratio, size_ratio,1]) @ Mratio @ np.diag([1 / size_ratio, 1 / size_ratio, 1])
            new_pts = cv2.perspectiveTransform(annots[i_match].reshape(-1,1,2), M).squeeze()

            #pitch_in = pitch[annots_idx[i_match]]
            pitch_in = annots[i_match]
            new_in = new_pts
            M2img, mask = cv2.findHomography(pitch_in, new_in, cv2.RANSAC)
            M2pitch, mask = cv2.findHomography(new_in, pitch_in, cv2.RANSAC)
            Hs.append(np.stack((M, M2img, M2pitch)))
        
        
        if plot_pts:        
            for pt in new_pts.astype(np.int16):
                cv2.circle(frame, pt, 10, (0,255,0), -1)
                
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"{i} {i_match}")
            plt.axis('off')
            plt.show()
        
        if i%100 == 0:
            print(f"{i} {time.time() - t0:.2f} temps de match {t_match} ")
    

Hs = np.array(Hs)
np.save(Hs_name, Hs)
Hs = np.load(Hs_name)
i_ref = Hs[:,0,2,2].copy().astype(np.int16)
Hs[:,0,2,2] = 1

avi_name =  videos_folder/ f"results.avi"
video_capture = cv2.VideoCapture()
if video_capture.open( video_in ):
    w, h = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, init_frame) 
    ret, frame = video_capture.read()
    
    fps_write = fps
    video_writer = cv2.VideoWriter(avi_name,
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     fps_write, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (0,0,255)
    thickness = 3
    lineType = 2

    for i in range(2_000):
        ret, frame = video_capture.read()
        i_match = i_ref[i]
        new_pts = cv2.perspectiveTransform(annots[i_match].reshape(-1,1,2),
                                           Hs[i,0]).squeeze()
        
        for pt in new_pts.astype(np.int16):
            cv2.circle(frame, pt, 10, (0,255,0), -1)
        
        cv2.putText(frame,f"{i}", (100,100), font, fontScale, fontColor, thickness, lineType)
        video_writer.write(frame)
        
    
    video_capture.release()

print('conversion')
#cmd_1 = f"ffmpeg -i {avi_name}  -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {video_box}"
cmd_1 = f"ffmpeg -v quiet -i {avi_name}  -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {video_out}"
os.system(cmd_1)
 
cmd_2 = f"rm {avi_name}"
os.system(cmd_2)

