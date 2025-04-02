#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 09:21:50 2024

@author:
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # 🔹 Force interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
from matplotlib.backend_bases import MouseButton
import cv2

# Global variable for ROI selection
roi = None

def select_roi(eclick, erelease):
    """ Callback function for selecting a region of interest (ROI). """
    global roi

    if eclick.xdata is None or erelease.xdata is None:
        print("Invalid selection. Please draw a rectangle within the image area.")
        return
    
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    roi = np.int16([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    print(f"ROI selected: {roi}")

def button_press_callback(event):
    """ Callback for mouse button press events. """
    global pts_local, image

    image_pt = image.copy()
    if event.inaxes is None:
        return
    
    h, w = image.shape[:2]
    ix, iy = event.xdata, event.ydata
    if ix < 1 or ix > w or iy < 1 or iy > h:  # Ignore clicks outside the image
        return
    
    new_pt = np.array([event.xdata, event.ydata]).reshape(1,2)
    
    if event.button is MouseButton.LEFT:
        pts_local = np.append(pts_local, new_pt, axis=0)
        
    if event.button is MouseButton.RIGHT and len(pts_local) > 0:
        to_delete = np.argmin(np.linalg.norm(pts_local - new_pt, axis=1))
        pts_local = np.delete(pts_local, to_delete, axis=0)

    for pt in pts_local:
        cv2.circle(image_pt, pt.astype(np.int16), 3, (0, 255, 0), -1)
    
    global imgplot, fig
    imgplot.set_data(image_pt)
    fig.canvas.draw_idle()

def terminate(event):
    """ Closes all windows and stops execution. """
    global finished
    finished = True
    plt.close('all')

def show_roi_selection(img):
    """ Function to handle ROI selection interactively """
    global roi

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(img)

    selector = RectangleSelector(
        ax, select_roi,
        useblit=True,
        button=[1, 3],
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )

    plt.title("Select ROI and close window to continue")

    # Use blocking `plt.show()` to keep the window open
    plt.show(block=True)

    if roi is None:
        print("Error: ROI was not selected. Please restart and try again.")
        sys.exit(1)

# 🔹 Load Image
frame_idx = [1000, 100000, 100170, 170040]
i_frame = [104700, 104700+75, 104700+75+35]
<<<<<<< HEAD
i = 2
=======
i = 1
>>>>>>> 69d6912 (feat: atualiza superpoint.py)
img_name = f"img_{i_frame[i]}.png"
pts_name = f"pts_dict_{i_frame[i]}.npy"

if not os.path.exists(img_name):
    print(f"Error: Image {img_name} not found.")
    sys.exit(1)

img = cv2.imread(img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_ori = img.copy()

# 🔹 Load or Initialize Annotations
if os.path.exists(pts_name):
    pts_dict = np.load(pts_name, allow_pickle=True).item()
    pts = pts_dict.get('pts', np.array([]).reshape(-1, 2))
else:
    print(f"Warning: {pts_name} not found. Creating a new annotation file.")
    pts_dict = {"pts": np.array([]).reshape(-1, 2), "ident": np.array([])}
    pts = np.array([]).reshape(-1, 2)

finished = False

# 🔹 Ensure ROI Selection Before Continuing
show_roi_selection(img)

# 🔹 Extract ROI
xt, yt, xb, yb = roi
print(f"Using ROI: {xt}, {yt}, {xb}, {yb}")

while not finished:
    rgb = img.copy()
    for pt in pts:
        cv2.circle(rgb, pt.astype(np.int16), 3, (0, 255, 0), -1)

    fig_all, ax = plt.subplots(figsize=(13, 7))
    ax.imshow(rgb)
    
    button_ax = plt.axes([0.8, 0.025, 0.1, 0.04])
    button = Button(button_ax, 'Terminate')
    button.on_clicked(terminate)

    plt.axis("off")
    plt.title("Close window when selection is correct")
    plt.show()

    if not finished:
        pts_local = np.array([]).reshape(-1, 2)
        
        fig, ax = plt.subplots(figsize=(16, 9))
        image = rgb[yt:yb, xt:xb]
        imgplot = plt.imshow(image)
        
        fig.canvas.mpl_connect('button_press_event', button_press_callback)
        
        plt.axis("off")
        plt.title("Left click: Add point, Right click: Remove point, Close window to finish")
        plt.show()

        if len(pts) == 0:
            pts = pts_local + np.array([xt, yt]).reshape(1, 2)
        else:
            pts = np.vstack((pts, pts_local + np.array([xt, yt]).reshape(1, 2)))

pts_dict['pts'] = pts.copy()
np.save(pts_name, pts_dict)

# 🔹 Load Identifiers
idents = pts_dict.get('ident', np.array([])).astype(np.int16)
annot = True

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
fontColor = (0, 0, 0)
thickness = 2

if len(idents) == len(pts):
    annot = False

    for ident, pt in zip(idents, pts.astype(np.int16)):
        cv2.circle(img, pt, 5, (255, 0, 0), -1)
        cv2.putText(img, f"{ident}", pt, font, fontScale, fontColor, thickness)

    plt.imshow(img)
    plt.axis('off')
    plt.show()

    if input("Nouvelle annotation? (y/n): ").strip().lower() == 'y':
        annot = True

if annot:
    new_pts = pts[len(idents):]
    for pt in new_pts:
        im_plot = img_ori.copy()
        cv2.circle(im_plot, pt.astype(np.int16), 5, (0, 255, 0), -1)
        
        plt.imshow(im_plot)
        plt.title("Close window and enter point identity on keyboard")
        plt.axis('off')
        plt.show()
        
        new_in = int(input("Enter point identity: "))
        idents = np.append(idents, new_in)

    pts_dict['ident'] = idents.astype(np.int16)
    np.save(pts_name, pts_dict)

    for ident, pt in zip(idents, pts.astype(np.int16)):
        cv2.circle(img, pt, 5, (255, 0, 0), -1)
        cv2.putText(img, f"{ident}", pt, font, fontScale, fontColor, thickness)

    plt.imshow(img)
    plt.axis('off')
    plt.title("Final annotation")
    plt.show()
