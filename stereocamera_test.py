import numpy as np
import cv2

capture = cv2.VideoCapture(2)   # 비디오 캡처, video 장치 번호
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # width
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height

# stereo = cv2.StereoBM_create(numDisparities=0, blockSize=9)
max_disparity = 128
stereo = cv2.StereoSGBM_create(0, max_disparity, 11)
disparity_ = np.zeros((240,320,3))

while True:
    ret, frame = capture.read() # camera 읽기
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgL = gray[:,:320]
    imgR = gray[:,320:]

    disparity = stereo.compute(imgL,imgR)
    disparity = stereo.compute(imgL, imgR)
    cv2.filterSpeckles(disparity, 0, 40, max_disparity)

    disparity = 255*((disparity - np.min(disparity))/(np.max(disparity) - np.min(disparity)))
    disparity_ = np.stack((disparity,)*3, axis=2)
    disparity_ = np.uint8(disparity_)
    
    final_frame = np.hstack([frame, disparity_]) # or vconcat for vertical concatenation
    cv2.imshow("VideoFrame", final_frame) # 영상 출력
    if cv2.waitKey(1) > 0: break    # 입력이 들어오면 종료

capture.release()   
cv2.destroyAllWindows()