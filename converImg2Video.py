import numpy as np
import cv2, os

def write_video(filelist, imgdir, filename, savedir):
    img = cv2.imread(imgdir + '/' + filelist[0])
    h, w, _ = img.shape                                     # 크기
    fps = 20                                                # fps
    output_filename = savedir + '/' + filename + '.avi'     # 저장파일이름 
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')                # 코덱

    out_avi = cv2.VideoWriter(output_filename, fourcc, fps, (w,h))  # video writer object

    for imgname in filelist:
        imgpath = imgdir + '/' + imgname
        img = cv2.imread(imgpath)
        if img is not None:
            cv2.imshow('img', img)
            out_avi.write(img)
            cv2.waitKey(1)

    out_avi.release()
    cv2.destroyAllWindows()

def main():
    savedir = 'datas/videos'            # 저장 경로
    imgdir = 'datas/images/imageframes' # 이미지 경로
    filename = 'output_video'           # 저장 파일 이름
    filelist = os.listdir(imgdir)       # 이미지 경로에 있는 파일 리스트
    filelist.sort()                     # 파일 리스트 정렬

    write_video(filelist, imgdir, filename, savedir)

if __name__ == "__main__":
    main()