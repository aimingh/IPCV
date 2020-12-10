import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb2gray(img):
    if len(img.shape)==3:
        img = np.float32(img)
        gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
        # https://gammabeta.tistory.com/391 참고
        # gray = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
        # 흑백 이미지로 만들 때 그냥 평균을 써도 된다.
        # 하지만 대부분의 알고리즘은 r,g,b 값에 각각 가중치를 줍니다.
        # 사람의 눈은 동일한 값을 가질 때 녹색이 가장 밝게 보이고 
        # R, B 순으로 밝게 보이기 때문입니다.
        return np.uint8(gray)
    else:
        return (img)

def gray2rgb(img):
    if len(img.shape)==2:
        rgb = np.stack((img,)*3,2)
        return rgb
    else:
        return img

path = 'lenna.png'
img_ = plt.imread(path)            # 원본영상
if img_.dtype != np.uint8
    img = np.uint8(255*img_)            

img_gray1 = rgb2gray(img)                                               # 흑백 영상으로 변환    (512,512,3)->(512,512)
img_gray2 = cv2.cvtColor(img[:,:,::-1], cv2.COLOR_BGR2GRAY)

img_rgb1 = gray2rgb(img_gray1)                                          # 변환된 흑백 영상을 3차원으로 칼라 스페이스에 맞도록 변환 (512,512)->(512,512,3)
img_rgb2 = cv2.cvtColor(img_gray2, cv2.COLOR_GRAY2BGR)                  
img_gray_diff = np.uint8(np.abs(np.int8(img_rgb2) - np.int8(img_rgb1))) # opencv의 방법과 직접 처리한 방법의 차이를 보기 위한 차영상

# display하기위해 결합하는 부분 concat
imgH1 = np.hstack((img, img_gray_diff))                             
imgH2 = np.hstack((img_rgb1, img_rgb2[:,:,::-1]))
imgFinal = np.vstack((imgH1, imgH2))

# plot
plt.figure()
plt.imshow(imgFinal)
plt.show()