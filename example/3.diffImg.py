import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt

def absdiff(img1, img2):
    img1 = np.int16(img1)
    img2 = np.int16(img2)
    result = np.abs(img1 - img2)
    return np.uint8(result)

def frame_diff(prev_frame, cur_frame):
    diff_frames = absdiff(prev_frame, cur_frame)
    diff_frames_with_cv = cv2.absdiff(prev_frame, cur_frame)

    final = np.hstack((cur_frame, diff_frames, diff_frames_with_cv))
    cv2.imshow('diff', final)

def main():
    cap = cv2.VideoCapture("datas/videos/Armbot.mp4")
    
    try:
        _, cur_frame = cap.read()
        _, prev_frame = cap.read()
        while True:
            frame_diff(prev_frame, cur_frame)
            
            prev_frame = cur_frame
            _, cur_frame = cap.read()
            if (cv2.waitKey(1) > 0) or (np.sum(cur_frame)==None): 
                break   
    except:
        print('Error: Exception')
    finally:
        cap.release()   
        cv2.destroyAllWindows()
        print('End...')

if __name__ == "__main__":
    main()