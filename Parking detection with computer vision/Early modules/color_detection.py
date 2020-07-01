import numpy as np
import cv2
img = cv2.imread("D:\CSE\Grad Project\Image based Slot detection\Parking images\Vertical - available/clh.jpg")
img = cv2.medianBlur(img, 5)
height, width = img.shape[:2]

def parking_slot_region(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_yellow (img):
    lower_range_yellow = np.array([22, 60, 200])
    upper_range_yellow = np.array([60, 255, 255])
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range_yellow, upper_range_yellow)
    output = cv2.bitwise_and(img, img, mask=mask)
    return mask, output

def detect_white (img):
    # HSV ranges for White
    lower_range_white = np.array([0, 0, 0])
    upper_range_white = np.array([0, 0, 255])
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range_white, upper_range_white)
    output = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("images", np.hstack([img, output]))
    cv2.imshow('mask', mask)

# Define the parking slot "Our region of interest"
parking_slot = [
    (0, height),
    (width/ 8, height/3 ),
    (7*width/8, height/3),
    (width, height)
]
cropped_image = parking_slot_region(
    img,
    np.array([parking_slot], np.int32),
)

mask_yellow, output_yellow =detect_yellow(cropped_image)
cv2.imshow("images", np.hstack([cropped_image, output_yellow]))
cv2.imshow('mask', mask_yellow)
cv2.waitKey(0)
cv2.destroyAllWindows()