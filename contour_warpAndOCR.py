from cv2 import cv2
import numpy as np
import test_pytesseract2 

def searchRect(img, contours, e=0.02):
    max_area = 0
    max_approx = 0
    
    for cnt in contours:
        epsilon = e*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx)==4:
            print(f'{cv2.contourArea(cnt)}')
            if max_area < cv2.contourArea(cnt):
                max_approx = approx
                max_area = cv2.contourArea(cnt)

    return max_approx

def wrapping(img, img_approx, wplus = 0):
    if len(img.shape)==3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    width = width + wplus
    pts1 = np.float32([img_approx[0][0], img_approx[3][0], img_approx[1][0], img_approx[2][0]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,matrix,(width,height))

def show_result(img, img_ocr):
    final = np.hstack((img, img_ocr))
    final = cv2.resize(final, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('result', final)
    cv2.waitKey()

def recognition(filename):
    img = cv2.imread(filename)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_bi = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(img_bi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_approx = searchRect(img, contours)
    img_wrap = wrapping(img_bi, img_approx, wplus=0)

    img_ocr, words = test_pytesseract2.get_OCR2(img_wrap)
    print(words['text'])
    img_wrap = cv2.cvtColor(img_wrap, cv2.COLOR_GRAY2BGR)
    show_result(img, img_ocr)

if __name__ == "__main__":
    filename = 'datas/images/namecard_02.jpg'
    recognition(filename)