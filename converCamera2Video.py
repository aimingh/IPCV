import numpy as np
import cv2, os

def camera2video():
    capture = cv2.VideoCapture(0)   # 비디오 캡처, video 장치 번호
    # w = int(capture.get(3))
    # h = int(capture.get(4))
    w = 1280
    h = 720
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)  # width
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h) # height

    savedir = 'datas/videos'            # 저장 경로
    filename = 'output_camera2video'           # 저장 파일 이름
    fps = 20 

    output_filename = savedir + '/' + filename + '.avi'
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out_avi = cv2.VideoWriter(output_filename, fourcc, fps, (w,h))  # video writer object
    output_filename = savedir + '/' + filename + '.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out_mp4 = cv2.VideoWriter(output_filename, fourcc, fps, (w,h))  # video writer object
    
    while True:
        ret, frame = capture.read() # camera 읽기
        if ret:
            out_avi.write(frame)
            out_mp4.write(frame)
            cv2.imshow('img', frame)
        if cv2.waitKey(1) > 0: break    # 키보드 입력이 들어오면 종료

    capture.release()       
    out_avi.release()
    out_mp4.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera2video()
    


