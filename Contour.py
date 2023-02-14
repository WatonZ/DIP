import cv2
import numpy as np

img = cv2.imread('sudoku-original.jpg')
img_gray = cv2.imread('sudoku-original.jpg',cv2.IMREAD_GRAYSCALE)

img_denoised = cv2.GaussianBlur(img_gray, (11, 11), 0)

thresh = cv2.adaptiveThreshold(img_denoised, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                cv2.THRESH_BINARY_INV, 5, 2)

kernel = np.ones((3,3),np.uint8)
thresh = cv2.dilate(thresh, kernel)  


outerbox = thresh
maxb = -1
maxPt = None

       
h, w = np.shape(outerbox)
for y in range(h):
    row = outerbox[y]
    for x in range(w):
        if row[x] >= 128:
            area = cv2.floodFill(outerbox, None, (x, y), 64)[0]
            if area > maxb:
                maxPt = (x, y)
                maxb = area

        
cv2.floodFill(outerbox, None, maxPt, (255, 255, 255))
for y in range(h):
    row = outerbox[y]
    for x in range(w):
        if row[x] == 64 and x != maxPt[0] and y != maxPt[1]:
            cv2.floodFill(outerbox, None, (x, y), 0)

kernel = np.ones((3,3), np.uint8)
outerbox = cv2.erode(outerbox, kernel)

lines = cv2.HoughLines(outerbox, 1, np.pi / 180, 200)

def drawLine(line, img):
    h, w = np.shape(img)
    if line[0][1] != 0:
        m = -1 / np.tan(line[0][1])
        c = line[0][0] / np.sin(line[0][1])
        cv2.line(img, (0, int(c)), (w, int(m * w + c)), 255)
    else:
        cv2.line(img, (int(line[0][0]), 0), (int(line[0][0]), h), 255)
    return img
drL = np.copy(outerbox)
for i in range(len(lines)):
    drimp = drawLine(lines[i], drL)
def mergeRelatedLines(lines, img):
    h, w = np.shape(img)
    for current in lines:
        if current[0][0] is None and current[0][1] is None:
            continue
        p1 = current[0][0]
        theta1 = current[0][1]
        pt1current = [None, None]
        pt2current = [None, None]
                
        if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
            pt1current[0] = 0
            pt1current[1] = p1 / np.sin(theta1)
            pt2current[0] = w
            pt2current[1] = -pt2current[0] / np.tan(theta1) + p1 / np.sin(theta1)
                
        else:
            pt1current[1] = 0
            pt1current[0] = p1 / np.cos(theta1)
            pt2current[1] = h
            pt2current[0] = -pt2current[1] * np.tan(theta1) + p1 / np.cos(theta1)
                
        for pos in lines:
            if pos[0].all() == current[0].all():
                continue
            if abs(pos[0][0] - current[0][0]) < 20 and  (pos[0][1] - current[0][1]) < np.pi * 10 / 180:
                p = pos[0][0]
                theta = pos[0][1]
                pt1 = [None, None]
                pt2 = [None, None]
                        
                if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
                    pt1[0] = 0
                    pt1[1] = p / np.sin(theta)
                    pt2[0] = w
                    pt2[1] = -pt2[0] / np.tan(theta) + p / np.sin(theta)
                        
                else:
                    pt1[1] = 0
                    pt1[0] = p / np.cos(theta)
                    pt2[1] = h
                    pt2[0] = -pt2[1] * np.tan(theta) + p / np.cos(theta)
                        
                if (pt1[0] - pt1current[0])**2 + (pt1[1] - pt1current[1])**2 < 64**2 and (pt2[0] - pt2current[0])**2 + (pt2[1] - pt2current[1])**2 < 64**2:
                    current[0][0] = (current[0][0] + pos[0][0]) / 2
                    current[0][1] = (current[0][1] + pos[0][1]) / 2
                    pos[0][0] = None
                    pos[0][1] = None
            
    return lines

     
lines = mergeRelatedLines(lines, outerbox)

topedge = [[1000, 1000]]
bottomedge = [[-1000, -1000]]
leftedge = [[1000, 1000]]
leftxintercept = 100000
rightedge = [[-1000, -1000]]
rightxintercept = 0
for i in range(len(lines)):
    current = lines[i][0]
    p = current[0]
    theta = current[1]
    xIntercept = p / np.cos(theta)
    if theta > np.pi * 80 / 180 and theta < np.pi * 100 / 180:
        if p < topedge[0][0]:
            topedge[0] = current[:]
        if p > bottomedge[0][0]:
            bottomedge[0] = current[:]


    if theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
        if xIntercept > rightxintercept:
            rightedge[0] = current[:]
            rightxintercept = xIntercept
        elif xIntercept <= leftxintercept:
            leftedge[0] = current[:]
            leftxintercept = xIntercept

       
drL= np.copy(outerbox)

drL = drawLine(leftedge, drL)
drL = drawLine(rightedge, drL)
drL = drawLine(topedge, drL)
drL = drawLine(bottomedge, drL)

leftedge = leftedge[0]
rightedge = rightedge[0]
bottomedge = bottomedge[0]
topedge = topedge[0]

        
left1 = [None, None]
left2 = [None, None]
right1 = [None, None]
right2 = [None, None]
top1 = [None, None]
top2 = [None, None]
bottom1 = [None, None]
bottom2 = [None, None]

if leftedge[1] != 0:
    left1[0] = 0
    left1[1] = leftedge[0] / np.sin(leftedge[1])
    left2[0] = w
    left2[1] = -left2[0] / np.tan(leftedge[1]) + left1[1]
else:
    left1[1] = 0
    left1[0] = leftedge[0] / np.cos(leftedge[1])
    left2[1] = h
    left2[0] = left1[0] - h * np.tan(leftedge[1])

if rightedge[1] != 0:
    right1[0] = 0
    right1[1] = rightedge[0] / np.sin(rightedge[1])
    right2[0] = w
    right2[1] = -right2[0] / np.tan(rightedge[1]) + right1[1]
else:
    right1[1] = 0
    right1[0] = rightedge[0] / np.cos(rightedge[1])
    right2[1] = h
    right2[0] = right1[0] - h * np.tan(rightedge[1])

bottom1[0] = 0
bottom1[1] = bottomedge[0] / np.sin(bottomedge[1])

bottom2[0] = w
bottom2[1] = -bottom2[0] / np.tan(bottomedge[1]) + bottom1[1]

top1[0] = 0
top1[1] = topedge[0] / np.sin(topedge[1])
top2[0] = w
top2[1] = -top2[0] / np.tan(topedge[1]) + top1[1]

        
        
leftA = left2[1] - left1[1]
leftB = left1[0] - left2[0]
leftC = leftA * left1[0] + leftB * left1[1]

rightA = right2[1] - right1[1]
rightB = right1[0] - right2[0]
rightC = rightA * right1[0] + rightB * right1[1]

topA = top2[1] - top1[1]
topB = top1[0] - top2[0]
topC = topA * top1[0] + topB * top1[1]

bottomA = bottom2[1] - bottom1[1]
bottomB = bottom1[0] - bottom2[0]
bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

                
detTopLeft = leftA * topB - leftB * topA

ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)

        
        
detTopRight = rightA * topB - rightB * topA

ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)

        
        
detBottomRight = rightA * bottomB - rightB * bottomA

ptBottomRight = ((bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)

        
        
detBottomLeft = leftA * bottomB - leftB * bottomA

ptBottomLeft = ((bottomB * leftC - leftB * bottomC) / detBottomLeft,
                               (leftA * bottomC - bottomA * leftC) / detBottomLeft)

       
cv2.circle(drL, (int(ptTopLeft[0]), int(ptTopLeft[1])), 5, 0, -1)
cv2.circle(drL, (int(ptTopRight[0]), int(ptTopRight[1])), 5, 0, -1)
cv2.circle(drL, (int(ptBottomLeft[0]), int(ptBottomLeft[1])), 5, 0, -1)
cv2.circle(drL, (int(ptBottomRight[0]), int(ptBottomRight[1])), 5, 0, -1)
        

leftedgelensq = (ptBottomLeft[0] - ptTopLeft[0])**2 + (ptBottomLeft[1] - ptTopLeft[1])**2
rightedgelensq = (ptBottomRight[0] - ptTopRight[0])**2 + (ptBottomRight[1] - ptTopRight[1])**2
topedgelensq = (ptTopRight[0] - ptTopLeft[0])**2 + (ptTopLeft[1] - ptTopRight[1])**2
bottomedgelensq = (ptBottomRight[0] - ptBottomLeft[0])**2 + (ptBottomLeft[1] - ptBottomRight[1])**2
maxlength = int(max(leftedgelensq, rightedgelensq, bottomedgelensq, topedgelensq)**0.5)

        
src = [(0, 0)] * 4
dst = [(0, 0)] * 4
src[0] = ptTopLeft[:]
dst[0] = (0, 0)
src[1] = ptTopRight[:]
dst[1] = (maxlength - 1, 0)
src[2] = ptBottomRight[:]
dst[2] = (maxlength - 1, maxlength - 1)
src[3] = ptBottomLeft[:]
dst[3] = (0, maxlength - 1)
src = np.array(src).astype(np.float32)
dst = np.array(dst).astype(np.float32)

extracted= cv2.warpPerspective(np.copy(img_gray), cv2.getPerspectiveTransform(src, dst), (maxlength, maxlength))

extracted =  cv2.GaussianBlur(extracted,(11,11),0)
thrh = cv2.adaptiveThreshold(extracted,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV,11,2)

cv2.imshow('result',thrh )
cv2.imwrite('output.jpg', thrh)

cv2.waitKey(0)
