import numpy as np
import matplotlib.pyplot as plt

path = 'lenna.png'
A = plt.imread(path)            # 원본영상)
box = [256,512,0,256]                        # cropping하기위한 박스 위치 y1,y2,x1,x2
mask = np.zeros((512,512,3))                # mask 초기화
mask[box[0]:box[1],box[2]:box[3],:] = 1   # box에 따른 mask 생성

A_mask = A*mask                         # 마스크 영역만 남긴다.

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,2,1)    
a1.imshow(A)       # 원본 영상
a2 = fig.add_subplot(2,2,2)
a2.imshow(mask)    # 마스크 영역
a3 = fig.add_subplot(2,2,3)
a3.imshow(A_mask)  # 마스크 영역만 남게 됨
a4 = fig.add_subplot(2,2,4)
a4.imshow(A_mask[box[0]:box[1],box[2]:box[3],:]) #잘린 영역만 표시
plt.show()