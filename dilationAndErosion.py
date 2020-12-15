from cv2 import cv2 as cv
import numpy as np

img = cv.imread("datas/images/lena.png")
imgCanny = cv.Canny(img,150,200)

kernel = np.ones((5,5),np.uint8)
imgDialation = cv.dilate(imgCanny,kernel,iterations=1)
imgEroded = cv.erode(imgDialation,kernel,iterations=1)

kernel = np.ones((5,9),np.uint8)
imgDialation1 = cv.dilate(imgCanny,kernel,iterations=1)
imgEroded1 = cv.erode(imgDialation1,kernel,iterations=1)

kernel = np.ones((9,5),np.uint8)
imgDialation2 = cv.dilate(imgCanny,kernel,iterations=1)
imgEroded2 = cv.erode(imgDialation2,kernel,iterations=1)

final = np.hstack((imgCanny, imgDialation, imgEroded))
final1 = np.hstack((imgCanny, imgDialation1, imgEroded1))
final2 = np.hstack((imgCanny, imgDialation2, imgEroded2))
cv.imshow("Canny Image 5 5",final)
cv.imshow("Canny Image 5 9",final1)
cv.imshow("Canny Image 9 5",final2)
cv.waitKey(0)
cv.destroyAllWindows()