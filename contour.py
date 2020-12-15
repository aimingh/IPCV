from cv2 import cv2 as cv
import numpy as np

def drawContours(img_rgb, contours):
    for cnt in contours:
        area = cv.contourArea(cnt)
        print(f'area of contours: {area}')
        cv.drawContours(img_rgb, [cnt], 0, (255,0,0), 1)
    # cv.drawContours(img_rgb, contours, 0, (255,0,0), 2)
    return img_rgb

def approximationContour(img, contours, e=0.02):
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        print(f'x: {x}, y: {y}, w: {w}, h: {h}')
        epsilon = e*cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        print(len(approx))
        cv.drawContours(img, [approx], 0, (0,255,255), 2)
    return img

def example():
    img_rgb = cv.imread('datas/images/shapes.png')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    _, img_bi = cv.threshold(img_gray, 127, 255, 0)

    contours, _ = cv.findContours(img_bi, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    img_rgb = drawContours(img_rgb, contours)
    img_approx = approximationContour(img_rgb, contours)
    
    cv.imshow("result", img_approx)
    cv.waitKey()

###############################################
## Try Detect shapes of objects with Contour ##
###############################################
def rectwithname(img, contours, e=0.02):
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        epsilon = e*cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        if len(approx)==3:
            cv.putText(img, "triangle", (x, y-5), cv.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0), 1)
        elif len(approx)==4:
            if w == h:
                cv.putText(img, "square", (x, y-5), cv.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0), 1)
            else:
                cv.putText(img, "rectangle", (x, y-5), cv.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0), 1)
        else:
            cv.putText(img, "circle", (x, y-5), cv.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 0), 1)
    return img

def try_Detect_shape_with_contour():
    img_rgb = cv.imread('datas/images/shapes.png')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)
    _, img_bi = cv.threshold(img_gray, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    contours, _ = cv.findContours(img_bi, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    img_approx = rectwithname(img_rgb, contours)

    cv.imshow("result", img_approx)
    cv.waitKey()

if __name__ == "__main__":
    # example() 
    try_Detect_shape_with_contour()