
# coding: utf-8

# In[320]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np
import math
import cv2


# In[321]:


# reading in an image
original = mpimg.imread('clg.jpg')
# Using median filter to remove any non-parking lot lines in the ground.
image = cv2.medianBlur(original,51)
plt.imshow(original)
plt.show()
height, width = image.shape[:2]
plt.imshow(image)
plt.show()


# In[322]:


# Fucntion to return the parking slot "Our region of interest"
def parking_slot_region(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255 
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# In[323]:


# Define the parking slot "Our region of interest"
parking_slot = [
    (0, height),
    (width/ 7, height/3 ),
    (6*width/7, height/3 ),
    (width, height)
]

cropped_image = parking_slot_region(
    image,
    np.array([parking_slot], np.int32),
)
plt.figure()
plt.imshow(cropped_image)
plt.show()


# In[324]:


# Convert to grayscale here.
gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
# Call Canny Edge Detection here.
cannyed_image = cv2.Canny(gray_image, 2, 30)
plt.figure()
plt.imshow(cannyed_image)
plt.show()


# In[325]:


# Removing the edges of the cropped photo
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cannyed_image = cv2.Canny(gray_image, 2, 15)
slot_edges = parking_slot_region(cannyed_image, np.array([parking_slot], np.int32))
plt.figure()
plt.imshow(slot_edges)
plt.show()


# In[356]:


#  Draw the detected hough lines to the original image
def draw_lines(img, lines, color=[0, 255 , 0], thickness=20):
    # If there are no lines to draw, exit.
    if lines is None:
        return
    # Make a copy of the original image.
    img = np.copy(img)
    # Create a blank image that matches the original in size.
    line_img = np.zeros((img.shape[0],img.shape[1],3), dtype=np.uint8,)
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img


# In[357]:


# Generating Lines using hough trasnformation
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image,2 ,30)
cropped_image = parking_slot_region(
    cannyed_image,
    np.array(
        [parking_slot],
        np.int32
    ),
)
lines = cv2.HoughLinesP(
    cropped_image, 
    rho=3, # distance resolution in pixels of the Hough grid
    theta=np.pi / 50 , # angular resolution in radians of the Hough grid
    threshold= 5,  # minimum number of votes (intersections in Hough grid cell)
    lines=np.array([]), 
    minLineLength= 50, # minimum number of pixels making up a line
    maxLineGap= 50 # maximum gap in pixels between connectable line segments
)
line_image = draw_lines(image, lines)
plt.figure()
plt.imshow(line_image)
plt.show()

