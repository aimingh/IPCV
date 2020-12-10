import numpy as np
import cv2, os

def writeFrame(videocapture, path, second, cnt):
    videocapture.set(cv2.CAP_PROP_POS_MSEC, second*1000)
    hasFrames, image = videocapture.read()
    if hasFrames:
        cv2.imwrite(path+"/image_"+str(cnt).zfill(3)+".png", image)
    return hasFrames

def main():
    # path
    directoryname = os.getcwd() + '/datas/images/imageframes'
    # filename = 'datas/videos/Armbot.mp4'
    filename = 'datas/videos/output_video.avi'

    # parameter
    sec = 0
    count = 0
    frameRate = 0.5

    if not os.path.isdir(directoryname):    # make directory if not exists
        os.mkdir(directoryname)

    cap = cv2.VideoCapture(filename)    # read video
    success = cap.isOpened()

    while success:
        sec = sec + frameRate   # we will get img og some point in video
        success = writeFrame(cap, directoryname, sec, count)    # write image 
        count = count + 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()