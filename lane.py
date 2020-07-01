import numpy as np
import cv2
img = cv2.imread("D:\CSE\Grad Project\Image based Slot detection\Integrated.png")
img = cv2.medianBlur(img, 5)
height, width = img.shape[:2]


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
    maskinv = (255-mask)
    a = np.empty_like(img)
    a.fill(190)
    maskinv = cv2.bitwise_and(maskinv, a)
    
    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    masked_image = cv2.bitwise_or(masked_image, maskinv)
    return masked_image


def detect_yellow (img):
    lower_range_yellow = np.array([22, 60, 200])
    upper_range_yellow = np.array([60, 255, 255])
    
    #kernel = np.ones((5,5),np.uint8)
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range_yellow, upper_range_yellow)
    #remove verticle lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask=cv2.erode(mask, kernel, iterations=1)
    #return verticle lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))    #mask=cv2.dilate(mask,kernel,iterations=3)
    edges=cv2.Canny(mask,100,200)
    #edges=cv2.dilate(edges,kernel,iterations=1)
    _, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(edges, contours, -1, (255,255,255), 3)
    output = cv2.bitwise_and(img, img, mask=mask)
    return mask, output,edges

def detect_white (img):
    # HSV ranges for White
    lower_range_white = np.array([0, 210, 0])
    upper_range_white = np.array([255, 255, 255])
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hsv, lower_range_white, upper_range_white)
    #remove verticle lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))  
    mask=cv2.erode(mask, kernel, iterations=1)
    #return verticle lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))    #mask=cv2.dilate(mask,kernel,iterations=3)
    edges=cv2.Canny(mask, 100, 200)
    edges=cv2.dilate(edges,kernel,iterations=1)
    _, contours, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    cv2.drawContours(edges, contours, -1, (255,255,255), 3)
    output = cv2.bitwise_and(img, img, mask=mask)
    return mask, output,edges

def getFourPoints (edges):
    y = edges.nonzero()[0]
    x = edges.nonzero()[1]
    
    
    y_min = y[np.argmin(y)]
    x_y_min = x[np.argmin(y)]
    
    y_max = y[np.argmax(y)]
    x_y_max = x[np.argmax(y)]
    
    #third point
    y_x_min = y[np.argmin(x)]
    x_min = x[np.argmin(x)]
    third = np.array([y_x_min, x_min])
    
    #last point
    y_x_max = y[np.argmax(x)]
    x_max = x[np.argmax(x)]
    forth = np.array([y_x_max, x_max])
    
    if abs(x_y_min-x_min) < abs(x_y_min-x_max):
        first = np.array([y_min, x_y_min])
        second=np.array([y_min, x_max-(abs(x_y_min-x_min))])
    else:
        first=np.array([y_min,x_min+(abs(x_y_min-x_max))])
        second=np.array([y_min,x_y_min])
    return first,second,third,forth

img=image_resize(img,600,600)
height, width = img.shape[:2]
mask_yellow, output_yellow,edges=detect_yellow(img)
mask_white, output_white,edges2=detect_white(img)
edges = cv2.bitwise_or(edges, edges2)

first,second,third,forth=getFourPoints(edges)
# Define the parking slot "Our region of interest"

parking_slot = [
    (0, height),
    (first[1], first[0] ),
    (second[1], second[0] ),
    (width, height)
]

cropped_image = parking_slot_region(
    img,
    np.array([parking_slot], np.int32),
)


#cv2.imshow("images", np.hstack([cropped_image, output_yellow,output_white]))
#cv2.imshow('img', img)

cv2.imshow('output', cropped_image)
#cv2.imshow('output_yellow', output_yellow)

#cv2.imshow('mask2', mask_white)
#cv2.imshow('edges', edges)
#cv2.imshow('edges2', edges2)




cv2.waitKey(0)
cv2.destroyAllWindows()