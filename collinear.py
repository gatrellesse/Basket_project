import numpy as np
import cv2
import matplotlib.pyplot as plt

i_frame = [104700, 104700+75, 104700+75+35]
i = 2
img_name = f"img_{i_frame[i]}.png"
pts_name = f"pts_dict_{i_frame[i]}.npy"
# Load the .npy file
data = np.load(pts_name, allow_pickle=True).item()
# Load the image
img = cv2.imread(img_name)
# Extract points and identities
pts = data["pts"]
pts_2 = pts.copy()
idents = data["ident"]

# Define collinear point groups
collinearH_Points = [[1, 3, 5], [19, 21], [7, 9, 13, 15], [11, 17, 23], [10, 16, 22], [6, 8, 12, 14], [18, 20], [0, 2, 4]]
collinearV_Points = [[1, 19, 7, 6, 18, 0], [9, 11, 10, 8], [3, 23, 22, 2], [15, 17, 16, 14], [5, 21, 13, 12, 20, 4]]

# Draw original points in green
for x, y in pts:
    cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 255, 0), -1)

#Function to fit a set of points
def fit_and_project_line(points):
    """Fit a line using PCA and project points onto the best fit line."""
    points = np.array(points)
    if len(points) < 2:
        return points, None  # Not enough points to fit a line

    # Compute the mean (center of mass)
    mean = np.mean(points, axis=0)

    # Perform PCA to get the principal direction
    cov_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Principal direction (dominant eigenvector)
    direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Project points onto the best-fit line
    projected_points = np.array([mean + np.dot(pt - mean, direction) * direction for pt in points])

    return projected_points, direction

#Function to fit a set of collinear points and att the original vector
def fiting_line(pointsIdent):
    collinearPoints = []
    originalIdx = []
    for idx, i in enumerate(idents):
        if i in pointsIdent:
            collinearPoints.append(pts[idx])
            originalIdx.append(idx)
    collinearPoints, direction = fit_and_project_line(collinearPoints)
    #print(collinearPoints, direction)
    for i, pt in zip(originalIdx,collinearPoints):
        pts[i] = pt
    #print(originalIdx)
    return collinearPoints, direction, originalIdx

line_to_be_traced = None

#Projecting the old points in the regression and plotting it
parallel_lines_H = []
original_idx_H = []
directions = []

for colPoints in collinearH_Points:
    line_to_be_traced, direction, idxs = fiting_line(colPoints)
    parallel_lines_H.append(line_to_be_traced)
    original_idx_H.append(idxs)
    if direction is not None:directions.append(direction)
    line_to_be_traced= np.array(line_to_be_traced, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [line_to_be_traced], isClosed=False, color=(72, 125, 0), thickness=2)
        

def make_parallel(lines, directions, original_idx):
    """Adjusts line2 to be parallel to line1 while keeping its midpoint fixed."""
    direction = np.mean(directions, axis = 0)
    for i, line in enumerate(lines): #use lines[0] as reference
        if len(line) < 2:
            continue
        mean = np.mean(line, axis=0)  # Get the midpoint of line2
        # Compute new points for line2 using the direction of line1
        new_line = np.array([mean + (pt - mean).dot(direction) * direction for pt in line])
        line_to_be_traced= np.array(new_line, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [line_to_be_traced], isClosed=False, color=(0, 0, 0), thickness=2)
        # Update pts with the adjusted second line
        for idx, pt in zip(original_idx[i], new_line):
            pts[idx] = pt

#This arrises problems related to 3D being represented 2D
#so, just adjusting the points to be colinear and after 
#using demography matrix works just fine
#make_parallel(parallel_lines_H, directions, original_idx_H)

parallel_lines_V = []
original_idx_V = []
directions = []

#When we make the vertical points colinear, it is gonna 
#desalign a bit the horizontal ones
#if we want to avoid this, we gotta get the intersection between both support line
#but this ends dislocating the point from its original points(corner) a bit too
#it is a matter of pro and con.
for colPoints in collinearV_Points:
    line_to_be_traced, direction, idxs = fiting_line(colPoints)
    parallel_lines_V.append(line_to_be_traced)
    original_idx_V.append(idxs)
    if direction is not None:directions.append(direction)
    line_to_be_traced= np.array(line_to_be_traced, np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [line_to_be_traced], isClosed=False, color=(0, 0, 255), thickness=2)


# Draw projected collinear points in red
img_write = 'arrests' + str(i_frame[i]) + '.jpg'
cv2.imwrite(img_write, img)

