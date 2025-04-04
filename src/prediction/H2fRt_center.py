#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:22:44 2024
https://github.com/SoccerNet/sn-calibration/blob/main/src/camera.py
Hs is (n_frame, 3, 3, 3)
Hs[:,0] homographies src_pts are anotations from annots[i_ref]
i_ref in Hs[0,:,2,2]
Hs[:,1] homographi src_pts are pitch points in pitch coordiantes
  des_pts are pts on screen point with screen coordinates
Hs[:2] the inverse of Hs[:,1] will be used to get players coordinates in pitch frame

Pour interpoler, le passage par KRt ne fait que rajouter de l'erreur
même en filtrant


@author: fenaux
"""
import numpy as np
import matplotlib.pyplot as plt

import os
import time
import cv2

from scipy.spatial.transform import Rotation as R


#source_video_path = 'data/2e57b9_0.mp4'

extract = 'supt2'
Hs_file = f"Hs_{extract}.npy"
fRt_file = f"fRt_{extract}.npy"

Hs = np.load(Hs_file)
i_ref = Hs[:,0,2,2].copy().astype(np.int16)
Hs [:,0,2, 2] = 1

homogs = Hs[:,1]



#frame.shape = (1080, 1920, 3)
cxy = np.array([960, 540])

#homogs[:,1] *= -1 # remet y positif vers le haut voir changeYsign
#cxy[1] *= -1


def KfromHcenter(homographies, cxy):
    """
    This method initializes the calibration matrix from the homography between the world plane of the pitch
    and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
    Multiple View Geometry in computer vision, p225). The extraction is sensitive to noise, which is why we keep the
    principal point in the middle of the image rather than using the one extracted by this method.
    :param homography: homography between the world plane of the pitch and the image
    """
    Tr_inv = np.eye(3)
    Tr_inv[:2,2] = -cxy
    
    n = len(homographies)
    Tr_inv = np.tile(Tr_inv, (n,1,1))
    
    Hc = Tr_inv @ homographies
    
    H = np.reshape(Hc, (-1,9))
    n = H.shape[0]
    A = np.zeros((n, 6, 6))
    A[:, 0, 1] = np.ones(n)  # w1 = 0
    A[:, 1, 0] = np.ones(n)  # w0 - w2 = 0
    A[:, 1, 2] = -np.ones(n)
    A[:, 2, 3] = 1 # w3 = 0
    A[:, 3, 4] = 1 # w4 = 0
    A[:, 4, 0] = H[:, 0] * H[:, 1]
    A[:, 4, 1] = H[:, 0] * H[:, 4] + H[:, 1] * H[:, 3]
    A[:, 4, 2] = H[:, 3] * H[:, 4]
    A[:, 4, 3] = H[:, 0] * H[:, 7] + H[:, 1] * H[:, 6]
    A[:, 4, 4] = H[:, 3] * H[:, 7] + H[:, 4] * H[:, 6]
    A[:, 4, 5] = H[:, 6] * H[:, 7]
    A[:, 5, 0] = H[:, 0] * H[:, 0] - H[:, 1] * H[:, 1]
    A[:, 5, 1] = 2 * H[:, 0] * H[:, 3] - 2 * H[:, 1] * H[:, 4]
    A[:, 5, 2] = H[:, 3] * H[:, 3] - H[:, 4] * H[:, 4]
    A[:, 5, 3] = 2 * H[:, 0] * H[:, 6] - 2 * H[:, 1] * H[:, 7]
    A[:, 5, 4] = 2 * H[:, 3] * H[:, 6] - 2 * H[:, 4] * H[:, 7]
    A[:, 5, 5] = H[:, 6] * H[:, 6] - H[:, 7] * H[:, 7]

    u, s, vh = np.linalg.svd(A)
    w = vh[:,-1]
    W = np.zeros((n, 3, 3))
    W[:, 0, 0] = w[:, 0] / w[:, 5]
    W[:, 0, 1] = w[:, 1] / w[:, 5]
    W[:, 0, 2] = w[:, 3] / w[:, 5]
    W[:, 1, 0] = w[:, 1] / w[:, 5]
    W[:, 1, 1] = w[:, 2] / w[:, 5]
    W[:, 1, 2] = w[:, 4] / w[:, 5]
    W[:, 2, 0] = w[:, 3] / w[:, 5]
    W[:, 2, 1] = w[:, 4] / w[:, 5]
    W[:, 2, 2] = w[:, 5] / w[:, 5]

    try:
        Ktinv = np.linalg.cholesky(W)
    except np.linalg.LinAlgError:
        K = np.eye(3)
        return False, K

    K = np.linalg.inv(np.transpose(Ktinv, (0,2,1)))
    K /= K[:, 2, 2].reshape(-1,1,1)
    K[:,:2,2] = cxy.reshape(1,2)

    return True, K

def get_fRt(homographies, Ks):
        """
        This method initializes the essential camera parameters from the homography between the world plane of the pitch
        and the image. It is based on the extraction of the calibration matrix from the homography (Algorithm 8.2 of
        Multiple View Geometry in computer vision, p225), then using the relation between the camera parameters and the
        same homography, we extract rough rotation and position estimates (Example 8.1 of Multiple View Geometry in
        computer vision, p196).
        :param homography: The homography that captures the transformation between the 3D flat model of the soccer pitch
         and its image.
        """

        focals = np.sqrt(Ks[:,0,0] * Ks[:,1,1])

        hprim = np.linalg.inv(Ks) @ homographies
        lambda1 = 1 / np.linalg.norm(hprim[...,0], axis=1, keepdims=True)
        lambda2 = 1 / np.linalg.norm(hprim[...,1], axis=1, keepdims=True)
        lambda3 = np.sqrt(lambda1 * lambda2)

        r0 = hprim[...,0] * lambda1
        r1 = hprim[...,1] * lambda2
        r2 = np.cross(r0, r1)

        R = np.stack( (r0.reshape(-1,3),
                      r1.reshape(-1,3),
                      r2.reshape(-1,3)), axis=2)
        u, s, vh = np.linalg.svd(R)
        R = u @ vh
        u_minus = u.copy()
        u_minus[...,2] *= -1
        R_minus =  u_minus @ vh
        det_neg = np.linalg.det(R) < 0
        R[det_neg] = R_minus[det_neg]

        t = - np.transpose(R, (0,2,1)) @ (hprim[...,2].reshape(-1,3,1) * lambda3.reshape(-1,1,1))
        return focals, R, t.squeeze()
    
def rotation_matrix_to_pan_tilt_roll(rotation):
    """
    Decomposes the rotation matrix into pan, tilt and roll angles. There are two solutions, but as we know that cameramen
    try to minimize roll, we take the solution with the smallest roll.
    :param rotation: rotation matrix
    :return: pan, tilt and roll in radians
    """
    orientation = np.transpose(rotation)
    first_tilt = np.arccos(orientation[2, 2])
    second_tilt = - first_tilt

    sign_first_tilt = 1. if np.sin(first_tilt) > 0. else -1.
    sign_second_tilt = 1. if np.sin(second_tilt) > 0. else -1.

    first_pan = np.arctan2(sign_first_tilt * orientation[0, 2], sign_first_tilt * - orientation[1, 2])
    second_pan = np.arctan2(sign_second_tilt * orientation[0, 2], sign_second_tilt * - orientation[1, 2])
    first_roll = np.arctan2(sign_first_tilt * orientation[2, 0], sign_first_tilt * orientation[2, 1])
    second_roll = np.arctan2(sign_second_tilt * orientation[2, 0], sign_second_tilt * orientation[2, 1])

    # print(f"first solution {first_pan*180./np.pi}, {first_tilt*180./np.pi}, {first_roll*180./np.pi}")
    # print(f"second solution {second_pan*180./np.pi}, {second_tilt*180./np.pi}, {second_roll*180./np.pi}")
    if np.fabs(first_roll) < np.fabs(second_roll):
        return first_pan, first_tilt, first_roll
    return second_pan, second_tilt, second_roll


t0 = time.time()
"""
status, Ks = KfromHcenter(homogs, cxy)
if status: valids = np.arange(len(homogs))
else: 
    Ks = []
    valids = []
    for i, homog in enumerate(homogs):
        status, K = KfromHcenter(homog, cxy)
        if status:
            Ks.append(K)
            valids.append(i)
    Ks = np.array(Ks).squeeze()
    valids = np.array(valids)

focs, Rs, Ts = get_fRt(homogs[valids], Ks)

# estimation of angles can also be made as 
# angles = R.from_matrix(Rs).as_euler('zxz') but with no garantee to have smallest roll
# the angles obtained are the opposite see minus sign on line 223
angles = []
for r in Rs:
    pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(r)
    angles.append([pan, tilt, roll])
angles = np.array(angles)
angles[:,0] = np.where(angles[:,0] > np.pi/2, angles[:,0] - np.pi, angles[:,0])

Ts_med = np.median(Ts[Ts[:,2]>0], axis=0)

KRt = np.column_stack(
    (focs, Ks[:,0,1], angles, np.tile(Ts_med,(len(Ks),1)))
    )
#np.save(KRt_file, KRt)

# Kalman filtering before interpolation ?

idxs = np.arange(len(homogs))
KRt_2 = KRt[::2,:5]

from sklearn.linear_model import RANSACRegressor
from scipy.linalg import expm, block_diag
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

reg = RANSACRegressor(random_state=0).fit(KRt_2[:,2:4], KRt_2[:,-1])
pred_roll = reg.predict(KRt_2[:,2:4])

# state transition matrix
# dim_x = 2 pour vitesse constante pendant l'intervalle, dimx = 3 accélération constante
# dimx = 4 pour jerk constant
dim_x = 2
A = np.zeros((dim_x,dim_x))
for i in range(dim_x-1): A[i, i+1] = 1
dt = 1#1/fs
Fkal = block_diag(expm(A * dt), expm(A * dt ), expm(A * dt), expm(A * dt )) # focal, pan tilt, roll

KalFi = KalmanFilter (dim_x=Fkal.shape[0], dim_z=4)
KalFi.F = Fkal

# Measurement fonction or matrix
Hkal = np.zeros((1,dim_x))
Hkal[0,0] = 1 # only poisition is measured
KalFi.H = block_diag(Hkal, Hkal, Hkal, Hkal)

mag_error = np.array([20, 0.005,  0.002, 0.002])
# mesasurement errors for hte differents measurements based on visual inspection
# initial conditions
Pdiag = np.ones(KalFi.P.shape[0])
Pdiag[::2] = 2*mag_error # arbitrary choice
# initial speed is left to zero and corresponding P values are big
Pdiag[1] = 10
KalFi.P = np.diag(Pdiag)

# Measurement noise matrix
# estimation de la position
mag_error = np.array([20, 0.005,  0.002, 0.002])
KalFi.R *= np.diag(mag_error ** 2)


#varQ  #erreur de process doit représenter la variation de position, de vitesse... de jerk suivant 
# le modèle choisi pour le premier joueur, en y devrait être 12e-3 (dux fois plus que x)
VarQs = 2 * np.quantile(np.abs(np.diff(np.diff(KRt_2, axis=0), axis=0)), 0.9, axis=0)
VarQs = np.delete(VarQs,1) # K[0,1] is ignored
Qs = []
for varQ in VarQs:
    Qs.append(Q_discrete_white_noise(dim=dim_x, dt=dt, var=varQ**2))

KalFi.Q = block_diag(Qs[0], Qs[1], Qs[2], Qs[3])

zs = KRt_2[:,[0,2,3,4]].reshape(len(KRt_2),KalFi.H.shape[0],1)
KalFi.x = np.zeros((Fkal.shape[0],1))
KalFi.x[::2] = zs[0]

mu, cov, _, _ = KalFi.batch_filter(zs)
xs, Ps, KsKal,_ = KalFi.rts_smoother(mu, cov)
print(f"{time.time() - t0:.2f}")

z_smooth = xs.squeeze()[:,::2]
err = np.abs(zs[...,0] - z_smooth)
med_all = np.median(err, axis=0)

#KRt_2[:,[0,2,3,4]] = z_smooth.copy()


from scipy.interpolate import make_interp_spline
spl = make_interp_spline(idxs[::2], KRt_2, k=1, axis=0)  # k=1: linear
# for k=1: bc_type can only be None (otherwise bc_type="natural")
KRt_interp = spl(idxs)

Ks_int = []
for i in idxs:
    f = KRt_interp[i,0]
    K_int = np.diag([f, f, 1])
    K_int[:2,2] = cxy
    K_int[0,1] = KRt_interp[i,1]
    Ks_int.append(K_int)
Ks_int = np.array(Ks_int)

rs_int = R.from_euler('zxz', -KRt_interp[:,2:5])
#rs_int = R.from_euler('zxz', -angles)
rt_int = -rs_int.apply(Ts_med)
Rts_int = rs_int.as_matrix()



Rts_int[...,2] = rt_int
Hs_int = Ks_int @ Rts_int
#Hs_int = Ks @ Rts_int
Hs_int /= Hs_int[:,2,2].reshape(-1,1,1)

"""
H = np.reshape(homogs, (-1,9))
n_frame = len(homogs)
idxs = np.arange(n_frame)
from scipy.interpolate import make_interp_spline
spl = make_interp_spline(idxs[::3], H[::3], k=1, axis=0)  # k=1: linear
# for k=1: bc_type can only be None (otherwise bc_type="natural")
H_interp = spl(idxs)
Hs_int = homogs.copy() #np.reshape(H_interp, (n_frame,3,3))

h = 1080
w = 1920
pitch = np.load('pitch.npy')
pitch_reshaped = pitch.reshape(-1,1,2)
errors = []
for homog_int, homog in zip(Hs_int, homogs):
    screen_pts = cv2.perspectiveTransform(pitch_reshaped, homog).squeeze()
    screen_pts_int = cv2.perspectiveTransform(pitch_reshaped, homog_int).squeeze()
    
    in_screen_x = (screen_pts[:,0] > 0) * (screen_pts[:,0] < w)
    in_screen_y = (screen_pts[:,1] > 0) * (screen_pts[:,1] < h)
    in_screen = in_screen_x * in_screen_y
    
    deltas = (screen_pts - screen_pts_int)[in_screen]
    deltas_norm = np.linalg.norm(deltas, axis=1)
    errors.append( [deltas_norm.mean(), np.median(deltas_norm), deltas_norm.max()])
errors = np.array(errors)
print(time.time() - t0)

truc += 2

size_ratio = 1
video_out = f"pitch_supt{size_ratio}_interp.mp4"
video_in = "../ffb/CFBB vs UNION TARBES LOURDES PYRENEES BASKET Men's Pro Basketball - Tactical.mp4"
avi_name = 'results.avi'
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

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    fontColor = (0,0,255)
    thickness = 3
    lineType = 2

    for i in range(2_000):
        ret, frame = video_capture.read()
        i_match = i_ref[i]
        new_pts = cv2.perspectiveTransform(pitch_reshaped,
                                           Hs_int[i]).squeeze()
        
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

# illustration des erreurs
ul = 10
i = 1
err_ul = errors[:,i] < ul
err2_ul = errors_2[:,i] < ul
plt.hist(errors[err_ul,i], bins=100)
plt.hist(errors_2[err2_ul,i], bins=100)
#plt.hist(np.column_stack((errors[err2_ul,i], errors_2[err2_ul,i])), bins=100)
plt.xlim(0,ul), 
plt.legend(['resize','2'])