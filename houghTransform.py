import numpy as np
import cv2

# Load the .npy file
data = np.load('pts_dict_104700.npy')
img = cv2.imread('img_104700.png')
print(data)
# Get image dimensions
height, width, channels = img.shape
print(f"Height:{height}, Width:{width}")
# Create a black binary image (all zeros)
binary_arr = np.zeros((height, width), dtype=np.uint8)

# Define a list of white pixel coordinates (x, y)

whitePixels = [(int(pts[0]), int(pts[1])) for pts in data ]  # Ensure integer indices
print(whitePixels)
# Set these pixels to white (255)
for x, y in whitePixels:
    if 0 <= x < width and 0 <= y < height:  # Avoid out-of-bounds errors
        binary_arr[y, x] = 255

cv2.imwrite('binary.png', binary_arr)

# Apply Hough Transform 
roTreshold = 1 ## Increase step size in pixels
thetaTreshold = np.pi / 90 # higher treshold--> More precise angle detection
minVotes = 5 #min number of votes to be considerated as a line
lines = cv2.HoughLines(binary_arr, roTreshold, thetaTreshold, minVotes)

if lines is not None:  # Ensure lines were detected
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('houghlines3.jpg', img)
else:
    print("No lines detected.")
