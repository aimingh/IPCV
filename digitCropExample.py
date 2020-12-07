import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0,0,0,0,0,0,0,0,0,0],    # 원본영상, 숫자2
            [0,0,1,1,1,1,1,1,0,0],
            [0,1,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1,1,0],
            [0,0,1,1,1,1,1,1,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0],])

box = [5,10,0,5]                        # cropping하기위한 박스 위치 y1,y2,x1,x2
mask = np.zeros((10,10))                # mask 초기화
mask[box[0]:box[1],box[2]:box[3]] = 1   # box에 따른 mask 생성

A_mask = A*mask                         # 마스크 영역만 남긴다.

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,2,1)    
a1.imshow(A, cmap='gray')       # 원본 영상
a2 = fig.add_subplot(2,2,2)
a2.imshow(mask, cmap='gray')    # 마스크 영역
a3 = fig.add_subplot(2,2,3)
a3.imshow(A_mask, cmap='gray')  # 마스크 영역만 남게 됨
a4 = fig.add_subplot(2,2,4)
a4.imshow(A_mask[box[0]:box[1],box[2]:box[3]], cmap='gray') #잘린 영역만 표시
plt.show()