import cv2
import numpy as np
import matplotlib.pyplot as plt

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


# Empty lot image
img = cv2.imread('available.png',1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying median filter to reduce noise
img= cv2.medianBlur(img, 21)
cv2.imshow('median', img)


# Getting the histogram of the empty lot

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram of available parking lot")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist)
plt.xlim([0, 256])


# A lot containing a car
img2 = cv2.imread('unavailable.png',1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = cv2.medianBlur(img2, 21)
cv2.imshow('median2', img2)

# getting the histogram of the un available lot

hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram of unavailable parking lot")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist2)
plt.xlim([0, 256])
# subtraction of the 2 image to indicate if there are objects in the lot
img3 = img - img2
cv2.imshow('diff1', img3)
hist3 = cv2.calcHist([img3], [0], None, [256], [0, 256])
plt.figure()
plt.title("Grayscale Histogram of subtraction")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(hist3)
plt.xlim([0, 256])


# compare the histograms of the 2 images the empty lot and the lot containing a car
similarity = cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL)
print(similarity)
if similarity > .65:
   print("Empty lot: Parking Available")
else:
   print("There's an object in the lot: Parking Unavailable           ")

x = cv2.compareHist(hist,hist,cv2.HISTCMP_CORREL)
print(x)



