import numpy as np
import cv2

def data2img(data, normalization = 1, histogram = 0):  # normalization, type castinf(uint8). gray2rgb
    if normalization == 1:
        data = cv2.normalize(src=data, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    data = np.uint8(data)
    if histogram == 1:
        data = cv2.equalizeHist(data)
    if len(data.shape) == 2:
        data = np.stack((data,)*3, axis=2)
    return data

def main():
    # video setting
    capture = cv2.VideoCapture(2)   # 비디오 캡처, video 장치 번호
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # width
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # height

    # model setting
    windowsize = 3
    left_matcher = cv2.StereoSGBM_create(minDisparity=0,blockSize=5,P1=8 * 3 * windowsize ** 2, P2=32 * 3 * windowsize ** 2,)
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

    while True:
        # input img
        ret, frame = capture.read() # camera 읽기
        h, w, c = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        imgL = gray[:,:int(w/2)]
        imgR = gray[:,int(w/2):]

        # compute 
        displ = left_matcher.compute(imgL, imgR)
        dispr = right_matcher.compute(imgR, imgL)
        displ = np.int16(displ)
        dispr = np.int16(dispr)
        filteredImg = wls_filter.filter(displ, imgL, None, dispr) 

        # change to show img 
        displ2 = data2img(displ)
        dispr2 = data2img(dispr)
        filteredImg1 = data2img(filteredImg, 0)
        filteredImg2 = data2img(filteredImg, 1, 1)

        # display
        final_frame = np.hstack([frame, filteredImg1]) # concatenation
        disps = np.hstack([displ2, dispr2, filteredImg2])   # concatenation
        final_frame = np.vstack([final_frame, disps])   # concatenation
        cv2.imshow("VideoFrame", final_frame) # 영상 출력

        if cv2.waitKey(1) > 0: break    # 키보드 입력이 들어오면 종료

    capture.release()   
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()