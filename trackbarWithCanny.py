from cv2 import cv2
import numpy as np

def nothing():
    pass

def sobel(img):
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    # sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    # sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    return laplacian

cv2.namedWindow("Canny Edge")
cv2.createTrackbar('low threshold', 'Canny Edge', 0, 1000, nothing)
cv2.createTrackbar('high threshold', 'Canny Edge', 0, 1000, nothing)
img_gray = cv2.imread('datas/images/shapes.png', cv2.IMREAD_GRAYSCALE)

while True:
    low = cv2.getTrackbarPos('low threshold', 'Canny Edge')
    high = cv2.getTrackbarPos('high threshold', 'Canny Edge')
    img_canny = cv2.Canny(img_gray, low, high)
    img_sobel = sobel(img_gray)

    finel = np.hstack((img_canny, img_sobel))
    cv2.imshow("Canny Edge", finel)
    if cv2.waitKey(1) == ord('q'):
        break