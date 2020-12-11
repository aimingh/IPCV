from cv2 import cv2
import time
import numpy as np

# haar cascade 기반 face detection
# img = cv2.imread("datas/images/people.jpg")
img = cv2.imread("datas/images/faces.jpg")

# detection model 
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
# smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 흑백영상 전환

faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴영상 검출
for (x,y,w,h) in faces: 
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)    # 찾아진 얼굴영상 위치 표시 (4각형 박스)

    # 눈 찾기
    face_img = img[y:y+h,x:x+w,:]
    eyes = eye_cascade.detectMultiScale(face_img, 1.3, 5)
    for (x1,y1,w1,h1) in eyes: 
        cv2.rectangle(img,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(0,255,0),2) 
    # smile = smile_cascade.detectMultiScale(face_img, 1.5, 10)
    # for (x1,y1,w1,h1) in smile: 
    #     cv2.rectangle(img,(x+x1,y+y1),(x+x1+w1,y+y1+h1),(0,0,255),2)   

img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
cv2.imshow("VideoFrame", img) # 영상 출력
cv2.waitKey()

cv2.destroyAllWindows()

