import cv2
import numpy as np


img = cv2.imread("C:/Users\Eslam.Wael\Desktop/Capture.PNG")
img = cv2.medianBlur(img, 5)
height, w = img.shape[:2]

point1 = np.float32([[0, 0], [400, 0], [0, 400], [400, 400]])
point2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])

P = cv2.getPerspectiveTransform(point1, point2)

output = cv2.warpPerspective(img, P, (300, 300))

cv2.imshow('output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()