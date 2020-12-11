import numpy as np
from cv2 import cv2

def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)

    # return_diff = cv2.bitwise_and(diff_frames_1, diff_frames_2)
    return_diff = cv2.absdiff(diff_frames_1, diff_frames_2)

    final = np.hstack((diff_frames_1, diff_frames_2, return_diff))
    cv2.imshow('img', final)

    threshold = len(return_diff[np.where(return_diff > 2)])
    if threshold > 500:
        print('threshold > 200 : ', threshold)
    return return_diff

def get_frame(cap, scaling):
    _, frame = cap.read()
    frame=cv2.resize(frame, None,fx=scaling,fy=scaling, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return gray

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # height
scaling_factor = 1
prev_frame = get_frame(cap, scaling_factor)
cur_frame = get_frame(cap, scaling_factor)
next_frame = get_frame(cap, scaling_factor)

while True:
    cv2.imshow('Object Movement', frame_diff(prev_frame, cur_frame, next_frame))
    prev_frame = cur_frame
    cur_frame = next_frame
    next_frame = get_frame(cap, scaling_factor)
    if cv2.waitKey(1) > 0: break    # 키보드 입력이 들어오면 종료

cap.release()   
cv2.destroyAllWindows()