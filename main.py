from skimage import io
import cv2 
import numpy as np
from scipy.ndimage import gaussian_filter

#read image
url = input('url:')
img = io.imread(url)

#kernel for opening and closing
kernel_o = np.ones((11,11),np.uint8)
kernel_c = np.ones((9,9),np.uint8)

#Inverse the gray image, and genetrate the color histogram 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
grayinv = 255-gray
hist = cv2.calcHist([grayinv], [0], None, [256], [0, 256])


masked = cv2.inRange(grayinv, 180,210)

#remove noise
closing = cv2.morphologyEx(masked, cv2.MORPH_CLOSE, kernel_c)
#remove noise
blur = gaussian_filter(closing, sigma=6)
thresh2 = cv2.threshold(blur,100,255, cv2.THRESH_BINARY)[1]
#cut thin connected area
opening = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN, kernel_o, iterations=2)

#Find contours and show the image
imageOK = opening
contours,hierarchy= cv2.findContours(imageOK,cv2.RETR_TREE,\
                                            cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(250,205,150),3)
io.imshow(img)
