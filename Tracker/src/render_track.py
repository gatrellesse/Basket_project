#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:49:57 2025

@author: fenaux
"""

import numpy as np
import supervision as sv
import cv2

from matplotlib import pyplot as plt

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.Color.BLUE,
    thickness=2
)
"""ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.DEFAULT,
    color_lookup='TRACK',
    thickness=2
)"""
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    thickness=2
)

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)

def plot_tracks(source_video_path: str, dict_file: str,
              target_video_path: str,
              track_kind: str = 'base',
              start: int = 0, end: int = -1):# -> Iterator[np.ndarray]:
    """
    après la détection des joueurs, modification pour ne rechercher les équipes que si
    une nouvelle trace est apparue

    """
    data_dict = np.load(dict_file, allow_pickle=True).item()
    bboxes_ = data_dict['bboxes']
    inframe = bboxes_[:,0].astype(np.int16)
    #frames = np.unique(inframe)
    bboxes = bboxes_[:,1:5]

    if track_kind == 'hmm': track_id = data_dict['track_ids_hmm'].astype(np.int16)
    elif track_kind == 'chain': track_id = data_dict['track_ids_chain'].astype(np.int16)
    elif track_kind == 'graph': track_id = data_dict['track_ids_graph'].astype(np.int16)
    else: track_id = data_dict['track_ids'].astype(np.int16)
    
    to_keep = data_dict['to_keep']
    inframe = inframe[to_keep]
    bboxes = bboxes[to_keep]
    track_id = track_id[to_keep]

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    if end == -1:
        end = video_info.total_frames
    source = sv.get_video_frames_generator(
        source_path=source_video_path, start=start, end=end)
    
    with sv.VideoSink(target_video_path, video_info) as sink:
        for i_frame, frame in enumerate(source):
            in_i_frame = np.where(inframe== i_frame)[0]
            boxes_in_frame = bboxes[in_i_frame]
            detections = sv.Detections(xyxy = boxes_in_frame,
                                          class_id=track_id[in_i_frame].astype(np.int16))#,
                                          #tracker_id = track_id[in_i_frame])
            labels = [str(tracker_id) for tracker_id in track_id[in_i_frame]]
            annotated_frame = frame.copy()
            annotated_frame = ELLIPSE_ANNOTATOR.annotate(
                annotated_frame, detections)
            annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
                annotated_frame, detections, labels=labels)
            
            cv2.putText(annotated_frame, str(i_frame), (100,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0,255,0))

            sink.write_frame(annotated_frame)