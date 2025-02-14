#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 17:11:17 2025
https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
@author: fenaux
"""

import time
import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

video_in = "../ffb/CFBB vs UNION TARBES LOURDES PYRENEES BASKET Men's Pro Basketball - Tactical.mp4"

MIN_MATCH_COUNT = 10
i_frame = [104700, 104700+75, 104700+75+35]

plot_pts = False
Hs_name = 'Hs2.npy'
video_out = 'pitch2.mp4'
size_ratio = 2

#Calculates the histogrames
def calc_ref_hist(imgs):
    return np.hstack([cv2.calcHist([img], [0], None, [256], [0, 256]) for img in imgs])

def best_match(new_img, ref_hist):
     new_hist = cv2.calcHist([new_img], [0], None, [256], [0, 256])

     match_probs = [cv2.matchTemplate(hist, new_hist, cv2.TM_CCOEFF_NORMED)[0][0] for hist in ref_hist.T]
     match_probs = np.array(match_probs)
     return np.argmax(match_probs), match_probs

def apply_sift(img, size_ratio=1):
    h, w = img.shape[:2]
    if size_ratio != 1:
        img = cv2.resize(img, (w//size_ratio, h//size_ratio))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    # keypoints = points coords
    # descriptors = feature vectors, matrix containing the info around the kp(gradient, edges, etc..)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray,None)
    return kp, desc

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
imgs=[]
annots = []
#loas all images and its annotations
for i in i_frame:
    annots_name = f"annots_{i}.npy"
    img = cv2.imread(f"img_{i}.png")
    imgs.append(img)
    annots.append(np.load(annots_name)[:,1:])

h, w = img.shape[:2]
ref_hist = calc_ref_hist(imgs)

kps = []
descs = []
#Apply sift for all imgs previously annotated
for img, annot in zip(imgs, annots):
    
    kp, desc = apply_sift(img, size_ratio=size_ratio)
    kps.append(kp)
    descs.append(desc)



init_frame = 100_000
avi_name = 'results.avi'
video_capture = cv2.VideoCapture()
#Opening and saving the video from a specific frame
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

if not os.path.exists(Hs_name):
    Hs = []
    t0 = time.time()
    for i in range(2_000):
    
        ret, frame = video_capture.read()
        #if i%10 != 0: continue
        
        i_match, probs = best_match(frame, ref_hist)
        
        #frame2 = cv2.resize(frame, (w//2, h//2))
        kp, desc = apply_sift(frame, size_ratio=size_ratio)
        
        #K-D-tree(3d binary tree) -->Best-bin-first-->priority queu(distance candidate<->query point)
        #flann-->nearest neighbors to the query point
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descs[i_match], desc,k=2)
     
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                #ratio = distance(m)/distance(n) -->m closest, n second closest
                good.append(m)
                
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kps[i_match][m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
         
        #Homography matrix to map SRC-->DST using Random Sample Consensus(RANSAC)
        #Select candidates images to apply the algo.
        #https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html

        Mratio, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        M =  np.diag([size_ratio, size_ratio,1]) @ Mratio @ np.diag([1 / size_ratio, 1 / size_ratio, 1])
        new_pts = cv2.perspectiveTransform(annots[i_match].reshape(-1,1,2), M).squeeze()
        M[2,2] = i_match
        Hs.append(M)
        
        #Plots the points found by the homography transformation
        for pt in new_pts.astype(np.int16):
            cv2.circle(frame, pt, 10, (0,255,0), -1)
        video_writer.write(frame)
    
        if plot_pts:        
            for pt in new_pts.astype(np.int16):
                cv2.circle(frame, pt, 10, (0,255,0), -1)
                
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title(f"{i} {i_match}")
            plt.axis('off')
            plt.show()
            
        if i%100 == 0:
            print(f"{i} {time.time() - t0:.2f} ")
        
    Hs = np.array(Hs)
    np.save(Hs_name, Hs)
    
    video_capture.release()

print('conversion')
#cmd_1 = f"ffmpeg -i {avi_name}  -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {video_box}"
cmd_1 = f"ffmpeg -v quiet -i {avi_name}  -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {video_out}"
os.system(cmd_1)
 
cmd_2 = f"rm {avi_name}"
os.system(cmd_2)


end_frame = 2_000
n_sample = 1
video_out = 'pitch.mp4'
Hs = np.load(Hs_name)
i_ref = Hs[:,2,2].astype(np.int16)
Hs[:,2,2] = 1
Hs_interp = Hs.copy()
avi_name = 'results.avi'
"""from scipy.interpolate import make_interp_spline
iHs = np.arange(0,end_frame, n_sample)
spl = make_interp_spline(iHs, Hs, k=1, axis=0)
i_frame = np.arange(end_frame)
Hs_interp = spl(i_frame)"""


init_frame = 100_000
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

#Using the HS interpolated we apply for each point in the image 
#from the pré annotated images by using perspective transformation
#It is basically a projective transformation
for i in range(end_frame):

    ret, frame = video_capture.read()
    i_match = i_ref[i]#[i//n_sample]
    
    new_pts = cv2.perspectiveTransform(annots[i_match].reshape(-1,1,2), Hs_interp[i]).squeeze()
    for pt in new_pts.astype(np.int16):
        cv2.circle(frame, pt, 10, (0,255,0), -1)
    video_writer.write(frame)
    if i%100==0: print(i)

video_capture.release()
video_writer.release()

print('conversion')
#cmd_1 = f"ffmpeg -i {avi_name}  -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {video_box}"
cmd_1 = f"ffmpeg -v quiet -i {avi_name}  -vf yadif=0 -vcodec mpeg4 -qmin 3 -qmax 3 {video_out}"
os.system(cmd_1)
 
cmd_2 = f"rm {avi_name}"
os.system(cmd_2)    