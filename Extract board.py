import cv2
import numpy as np

img = cv2.imread('digits.png')
img_gray = cv2.imread('digits.png',0)

img_blur =  cv2.GaussianBlur(img_gray,(9,9),1)

ret, thresh = cv2.threshold(img_blur,77, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((2,2),np.uint8)
opening = cv2.dilate(thresh, kernel)
opening = cv2.erode(opening, kernel)

contours, hierarchy = cv2.findContours(image=opening, mode=cv2.RETR_TREE, 
							method=cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
	cv2.rectangle(img,(x,y),(x+w,y+h),(0,255, 0),2)

cv2.imshow('Drawed contours', img) 
cv2.imwrite('mydigits.jpg',img)

cv2.waitKey(0)
