import cv2, time
import numpy as np

# haar cascade 기반 face detection
capture = cv2.VideoCapture(2)   # 비디오 캡처, video 장치 번호
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height

# detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = capture.read() # camera 읽기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 흑백영상 전환
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # 얼굴영상 검출
    for (x,y,w,h) in faces: 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)    # 찾아진 얼굴영상 위치 표시 (4각형 박스)
    cv2.imshow("VideoFrame", frame) # 영상 출력
    if cv2.waitKey(1) > 0: break    # 입력이 들어오면 종료

capture.release()
cv2.destroyAllWindows()
