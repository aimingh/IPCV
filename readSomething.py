from cv2 import cv2
import numpy as np

image = cv2.imread("datas/images/lena.png")             # read image file
video = cv2.VideoCapture("datas/videos/Armbot.mp4")     # read video file
webcam = cv2.VideoCapture(0)                            # read video device, /dev/video0

title = 'end: q, Image: 1, Video: 2, Webcam: 3, All: 4' # title of window
frameWidth = 640; frameHeight = 480
webcam.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)        # set width of video device
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)      # set height of video device

try:
    while True:
        success, frame_video = video.read()             # read a frame of video
        ret, frame_cam = webcam.read()                  # read a frame of webcam

        # resize size of image and video to size of webcam 
        frame_video = cv2.resize(frame_video, (frameWidth, frameHeight))
        image = cv2.resize(image, (frameWidth, frameHeight))
        # concatenation horizontaly
        frame = np.hstack((image, frame_video, frame_cam))

        cv2.imshow(title, frame)                        # show frame
        if cv2.waitKey(1) == ord('q'):
            break
except :
    print('Error: Exception')
finally:
    video.release()                                     # release video
    webcam.release()                                    # release webcam
    cv2.destroyAllWindows()                             # close all window
    print('End....')