# Project Title

## Overview

This project involves the extraction and transformation of key points from images, comparison of histograms, frame extraction from a video, applying a nearest neighbor algorithm, and the application of homography and perspective transformation. The final result is a video where the transformed points are saved in a new MP4 file.

## Flowchart for the Process

The following steps outline the flow of the process:

1. **Pre-annotation of Key Points**  
   Begin by identifying key points in images that will be tracked throughout the process.

2. **Calculate Histogram of Image**  
   Use an image’s pixel intensity distribution (histogram) to represent its content.

3. **Calculate the Best Match**  
   Compare the histogram of the new image with the old image’s histogram to identify the best match.

4. **Extract Specific Frames from Video**  
   Select specific frames from a video based on a given index or condition.

5. **Apply Nearest Neighbor Algorithm**  
   Find corresponding points between the old and new images by applying the nearest neighbor algorithm to the detected features.
   - Find KeyPoints and Features Describers: 
     * SIFT : Transformer that uses scale-invariant with Best-Bin-Search(based on K-D-tree).
     * Superpoint: CNN that uses pre-labed images to train.
     * Akaze: Uses non-linear diffusion to improve scale-invariant keypoints.

6. **Calculate Homography**  
   Calculate a homography matrix that maps points in the old image to the new image using methods like RANSAC.

7. **Apply Perspective Transformation**  
   Use the homography matrix to transform (map) key points from the old image’s coordinate system to the new one.

8. **Save the Result Points in an Output MP4 File**  
   Finally, save the transformed points and the updated frames into a new MP4 video file.

## Requirements

```bash
pip install opencv-python numpy transformers torch
