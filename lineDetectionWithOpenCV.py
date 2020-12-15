from cv2 import cv2
import numpy as np

def makeSimpleDistanceMap(bi_img):
    v = np.linspace(0, bi_img.shape[0]-1, bi_img.shape[0], dtype=np.int32)
    h = np.linspace(0, int(bi_img.shape[1]/2-1), int(bi_img.shape[1]/2), dtype=np.int32)

    vmap = np.hstack((v.reshape(v.shape[0],1),)*bi_img.shape[1])
    hmap = np.hstack((h,int(bi_img.shape[1]/2-1)-h))
    hmap = np.vstack((hmap,)*bi_img.shape[0])

    d_mask = hmap + vmap 
    d_img = d_mask * bi_img
    d_img =  d_img/d_img.max()
    return np.uint8(255 *d_img)

def bg_remove(cur_frame):
    _, frame_th_otsu = cv2.threshold(cur_frame,127,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    frame_di = cv2.erode(frame_th_otsu, kernel, iterations=4)
    # frame_di = cv2.dilate(frame_th_otsu, kernel, iterations=3)
    return frame_di

def linedetection(cur_frame):
    cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    mean = np.average(cur_frame)
    kernel = np.array([[0,1,0],
                        [1,1,1],
                        [0,1,0]], np.uint8)
    # frame_canny = cv2.Canny(cur_frame, 50, 150)
    _, frame_th = cv2.threshold(cur_frame,mean,255,cv2.THRESH_BINARY)
    
    frame_th_Dil = cv2.dilate(frame_th, kernel, iterations=3)
    frame_th_Ero = cv2.erode(frame_th_Dil,kernel,iterations=3)

    frame_th_Ero.shape
    frame_dist = makeSimpleDistanceMap(frame_th_Ero)

    _, frame_dist_th = cv2.threshold(frame_dist,200,255,cv2.THRESH_BINARY)
    frame_dist_th[:int(frame_dist_th.shape[0]/2)]=0

    frame_th_otsu = bg_remove(cur_frame)
    return frame_dist

def maskColorization(img, mask):
    loc = np.where(mask==255)
    loc1 = (loc[0], loc[1], 2*np.ones((loc[0].shape[0]), np.int64)) 
    img[loc1] = 255
    return img

def main():
    filepath = 'datas/videos/roadway_01.mp4'
    capture = cv2.VideoCapture(filepath)
    try:
        while True:
            _, cur_frame = capture.read()
            
            mask = linedetection(cur_frame)
            mask_frame = maskColorization(cur_frame, mask)

            cv2.imshow('result', mask)
            # cv2.waitKey()
            if cv2.waitKey(10) == ord('q'):
                break
    except:
        print('End...')
    finally:
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()