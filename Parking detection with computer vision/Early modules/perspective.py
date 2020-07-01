import cv2
import numpy as np

# read and show test image
img = cv2.imread("C:/Users\Eslam.Wael\Desktop/Capture.PNG")
height, width = img.shape[:2]

# Source points
top_left = [0, 0]
top_right = [width/2, 0]
bottom_right = [width/2, height/2]
bottom_left = [0, height/2]
pts = np.array([bottom_left, bottom_right, top_right, top_left], dtype=float)

# Target points

y_off = 100  # Y offset
top_left_dst = [width/2, 0]
top_right_dst = [0, 0]
bottom_right_dst = [width/8, height/2]
bottom_left_dst = [3*width/8, height/2]
dst_pts = np.array([bottom_left_dst, bottom_right_dst, top_right_dst, top_left_dst])

# Generate a preview to show where the warped bar would end up
preview = np.copy(img)
cv2.polylines(preview, np.int32([pts]), True, (0, 255, 0), 2)
cv2.polylines(preview, np.int32([dst_pts]), True, (0, 0, 0), 2)
cv2.imshow("preview", preview)

# calculate transformation matrix
pts = np.float32(pts.tolist())
dst_pts = np.float32(dst_pts.tolist())
M = cv2.getPerspectiveTransform(pts, dst_pts)

# wrap image and draw the resulting image
image_size = (img.shape[1], img.shape[0])
warped = cv2.warpPerspective(img, M, dsize= image_size, flags= cv2.INTER_LINEAR)
cv2.imshow("warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()