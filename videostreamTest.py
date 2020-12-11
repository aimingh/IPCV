from cv2 import cv2, time
import numpy as np

# opencv video 입출력
capture = cv2.VideoCapture(0)   # 비디오 캡처
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height

while True:
    ret, frame = capture.read() # camera 읽기
    cv2.imshow("VideoFrame", frame) # 영상 출력
    if cv2.waitKey(1) > 0: break    # 입력이 들어오면 종료

capture.release()
cv2.destroyAllWindows()
