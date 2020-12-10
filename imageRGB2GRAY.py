import numpy as np
import matplotlib.pyplot as plt
import cv2

def rgb2gray(img):
    if len(img.shape)==3:
        img = np.float32(img)
        # gray = (img[:,:,0] + img[:,:,1] + img[:,:,2])/3
        gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])
        # https://gammabeta.tistory.com/391 참고
        # 사람의 눈은 동일한 값을 가질 때 녹색이 가장 밝게 보이고 
        # R, B 순으로 밝게 보이기 때문에 가중치를 준다.
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
img = np.uint8(255*img_)
# img = cv2.imread(path)
# img = img[:,:,::-1]

img_gray1 = rgb2gray(img)
img_gray2 = cv2.cvtColor(img[:,:,::-1], cv2.COLOR_BGR2GRAY)

img_rgb1 = gray2rgb(img_gray1)
img_rgb2 = cv2.cvtColor(img_gray2, cv2.COLOR_GRAY2BGR)
img_gray_diff = np.uint8(np.abs(np.int8(img_rgb2) - np.int8(img_rgb1)))

imgH1 = np.hstack((img, img_gray_diff))
imgH2 = np.hstack((img_rgb1, img_rgb2[:,:,::-1]))
imgFinal = np.vstack((imgH1, imgH2))

# plot
plt.figure()
plt.imshow(imgFinal)
plt.show()