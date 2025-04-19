#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 17:16:45 2025

@author: fenaux
"""

import numpy as np
import cv2

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

draw_pitch = True

pts = [[0,0], [0,15],[14,0],[14,15],[28,0],[28,15]]
pts_raq = [[0, 7.5 - 2.45], [0, 7.5 + 2.45],
           [5.8, 7.5 - 2.45], [5.8, 7.5 + 2.45],
           [5.8, 7.5 - 1.8], [5.8, 7.5 + 1.8]]
pts_banc = [[8.325, 0], [8.325, 15]]
circle_med = [[14, 7.5 - 1.8], [14, 7.5 + 1.8]]
pts_raq = np.array(pts_raq)
pts_raq_sym = pts_raq.copy()
pts_raq_sym[:,0] = 28 - pts_raq[:,0]
pts_banc = np.array(pts_banc)
pts_banc_sym = pts_banc.copy()
pts_banc_sym[:,0] = 28 - pts_banc[:,0]
pts_3 = [[0, 0.9], [0, 15 - 0.9]]#, [2.99, 0.9], [2.99, 15 - 0.9],]
pts_3 = np.array(pts_3)
pts_3_sym = pts_3.copy()
pts_3_sym[:,0] = 28 - pts_3[:,0]
bask = (1.2 + 0.375, 7.5)

pitch = np.vstack((pts, pts_raq, pts_raq_sym, pts_3, pts_3_sym, circle_med, pts_banc, pts_banc_sym))
#pitch -= np.array([14,7.5]).reshape(1,2) # ne peremt pas de résoudre
#  le problème du point 0 renvoyé à droite pour la photo 0
pitch_lines = [[0,18],[18, 6],[6,7],[7,19],[19,1],
               [4,5],
               [8,10], [10,11], [11,9],
               [14,16], [16,17], [17,15],
               [6,8], [7,9], [12,14], [13,15],
               [0,2], [2,4], [1,3],[3,5],
               [2,22], [3,23]]
"""pitch_lines = [[0,18],[18, 6],[6,7],[7,19],[19,1],
               [4,20],[20,12],[12,13],[13,21],[21,5],
               [8,10], [10,11], [11,9],
               [14,16], [16,17], [17,15],
               [6,8], [7,9], [12,14], [13,15],
               [2,4], [1,3],[3,5],
               [2,22], [3,23]]"""
"""
circle_med = plt.Circle((14,7.5), 1.8, fill=False)
#wegdge_raq =  mpatches.Wedge((5.8, 7.5), 1.8, 270, 90, fill=False)
arc_raq = mpatches.Arc((5.8, 7.5), 2*1.8, 2*1.8, theta1=270, theta2=90)
teta_r = np.arcsin((7.5 - .9) / 6.75) * 180 / np.pi
arc_3 = mpatches.Arc(bask, 2*6.75, 2*6.75, theta1=360-teta_r, theta2=teta_r)
ax = plt.gca()
ax.plot(pitch[:,0], pitch[:,1],'.')
ax.add_patch(circle_med)
ax.add_artist(arc_raq)
ax.add_artist(arc_3)
plt.axis('equal')
plt.show()"""

if draw_pitch:
    plt.plot(pitch[:,0], pitch[:,1],'.')
    fig, ax = plt.subplots()
    ax.scatter(pitch[:,0], pitch[:,1])
    
    for line in pitch_lines:
        plt.plot(pitch[line][:,0],pitch[line][:,1], c='r')
    
    for i, pt in enumerate(pitch):
        ax.annotate(i, (pt[0], pt[1]))
    plt.show()
truc += 2
frame_idx = [1000, 100000, 100170, 170040]

i_frame = [104700, 104700+75, 104700+75+35]
i = 0
img_name = f"img_{i_frame[i]}.png"
pts_name = f"pts_{i_frame[i]}.npy"
annots_name = f"annots_{i_frame[i]}.npy"

img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_ori = img.copy()

pts = np.load(pts_name)
idents = pts[:,0].astype(np.int16)
pts = pts[:,1:]

src_pts = pitch[idents]
dst_pts = pts.copy()

H, mask = cv2.findHomography(src_pts, dst_pts)
rep_pts = cv2.perspectiveTransform(src_pts.reshape(-1,1,2), H).squeeze()
rep_error = np.linalg.norm(rep_pts - dst_pts, axis=1)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0,0,0)
thickness = 2
lineType = 2

for ident, pt in zip(idents, rep_pts.astype(np.int16)):
    cv2.circle(img, pt, 5, (255,0,0), -1)
    cv2.putText(img,f"{ident}", pt, font, fontScale, fontColor, thickness, lineType)
plt.imshow(img)
plt.axis('off')
plt.show()

img = img_ori.copy()
all_pts = cv2.perspectiveTransform(pitch.reshape(-1,1,2), H).squeeze()
pts_int = all_pts.astype(np.int16)
for i_pt, pt in enumerate(pts_int):
    cv2.circle(img, pt, 5, (0,255,0), -1)
    cv2.putText(img,f"{i_pt}", pt, font, fontScale, fontColor, thickness, lineType)
for line in pitch_lines:
    cv2.line(img, pts_int[line[0]], pts_int[line[1]], (255,0,0), 2)
plt.imshow(img)
plt.axis('off')
plt.show()

h, w = img.shape[:2]
in_img = (all_pts[:,1] < h) * (all_pts[:,1] > 0) * (all_pts[:,0] < w) * (all_pts[:,0] > 0)

#annots_idents = np.where(in_img)[0]
#annots_pts = all_pts[in_img]
#annots = np.column_stack((in_img, all_pts))
#np.save(annots_name, annots)
"""
from skimage import transform, measure
model = transform.ProjectiveTransform
trfm = measure.ransac((src_pts, dst_pts), model, min_samples=3,
        residual_threshold=5, max_trials=100)"""

from scipy.spatial.transform import Rotation as R
def makeH (forK, R, T, cxy):
    f, s = forK
    K = np.diag([f, f, 1])
    K[0,1] = s
    K[:2,2] = cxy
    
    rotT = - R @ T.reshape(3,1) 
    rt = R.copy()
    rt[:,2] = rotT.squeeze()
    calcH = K @ rt
    return calcH / calcH[2,2]

Ts_med = np.array([13.78358584, -4.44329415,  6.82521139])
r0 = np.array([[-0.82880854,  0.55884643,  0.02769598],
       [ 0.23622367,  0.39434887, -0.88808071],
       [-0.50722261, -0.72950644, -0.45885246]])
K0 = np.array([[ 1.23365333e+03, -2.28993103e+01,  9.60000000e+02],
        [ 0.00000000e+00,  1.23356517e+03,  5.40000000e+02],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

q0 = R.from_matrix(r0).as_euler('zxz')
q0p = q0.copy()
q0p[0] += np.pi
r0p = R.from_euler('zxz',q0p).as_matrix()
cxy = np.array([960, 540])

#makeH([K0[0,0], K0[0,1]], r0p, Ts_med, cxy)

def err_rep(forK, R, T, cxy, src_pts, dst_pts):
    Hcalc = makeH (forK, R, T, cxy)
    rep_pts = cv2.perspectiveTransform(src_pts.reshape(-1,1,2), Hcalc).squeeze()
    return np.linalg.norm(rep_pts - dst_pts, axis=1) ** 2
    
from scipy.optimize import leastsq

x0 = [np.sqrt(K0[0,0]*K0[1,1]), K0[0,1]]
args = (r0p, Ts_med, cxy, src_pts, dst_pts)

params = leastsq(err_rep, x0=x0, args=args)
