from cv2 import cv2
import numpy as np
import test_pytesseract2 

def example():
    img = cv2.imread("datas/images/cards.jpg")
    cv2.imshow("Image",img)

    # make transformation matrix 
    width,height = 250,350
    pts1 = np.float32([[111,219],[287,188],[154,482],[352,440]])
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)

    imgOutput = cv2.warpPerspective(img,matrix,(width,height))

    cv2.imshow("Output",imgOutput)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def wrapping(img, pts):
    height, width, _ = img.shape
    pts1 = pts
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,matrix,(width,height))

def preprocessing(rgbimg):
    img_gray = cv2.cvtColor(rgbimg, cv2.COLOR_BGR2GRAY)
    img_gray = (255-img_gray)
    # img_blur = cv2.GaussianBlur(img_gray,(31,31),0)
    # img_sharp = np.float32(img_gray) + 4*(np.float32(img_gray) - np.float32(img_blur))

    # # img_sharp[np.where(img_sharp>255)]=255
    # img_sharp = cv2.normalize(img_sharp,  img_sharp, 0, 255, cv2.NORM_MINMAX)
    return img_gray, img_gray, img_gray

def try_wrap():
    img = cv2.imread("datas/images/namecard_01.jpg")
    pts = np.float32([[400,464],[934,544],[96,1174],[960,1328]])

    img_wrap = wrapping(img, pts)
    img_gray, _, _ = preprocessing(img_wrap)
    
    img_gray2 = cv2.resize(img_gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    img_th = cv2.adaptiveThreshold(img_gray2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,-10)
    # ret,img_th = cv2.threshold(img_sharp,127,255,cv2.THRESH_BINARY)
    img_ocr, words = test_pytesseract2.get_OCR2(255-img_th)

    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # img_blur = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)
    # img_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_GRAY2BGR)
    img_th = cv2.cvtColor(img_th, cv2.COLOR_GRAY2BGR)
    final1 = np.hstack((img_wrap, img_gray ))
    final2 = np.hstack((img_th, img_ocr ))

    final1 = cv2.resize(final1, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    final = np.vstack((final1, final2))

    cv2.imshow('img1', final1)
    cv2.imshow('img2', final2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def try_wrap2():
    img = cv2.imread("datas/images/licenseplate_04.jpg")
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    pts = np.float32([[200,232],[467,272],[48,587],[480,664]])

    img_wrap = wrapping(img, pts)
    img_gray, img_blur, img_sharp = preprocessing(img_wrap)
    img_th = cv2.adaptiveThreshold(img_sharp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)
    img_ocr, words = test_pytesseract2.get_OCR2(img_th)

    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_blur = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)
    img_sharp = cv2.cvtColor(img_sharp, cv2.COLOR_GRAY2BGR)
    img_th = cv2.cvtColor(img_th, cv2.COLOR_GRAY2BGR)
    final1 = np.hstack((img_wrap, img_gray, img_blur ))
    final2 = np.hstack((img_sharp, img_th, img_ocr ))
    final = np.vstack((final1, final2))

    cv2.imshow('img1', img)
    # cv2.imshow('img2', final2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try_wrap()
