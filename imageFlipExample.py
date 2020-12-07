import numpy as np
import matplotlib.pyplot as plt

path = 'lenna.png'
A = plt.imread(path)            # 원본영상
B = np.identity(n=512)[::-1]
C_xflip = np.zeros((512,512,3))
C_yflip = np.zeros((512,512,3))
C_xyflip = np.zeros((512,512,3))

for i in range(3):  # 컬러영상은 3차원(RGB)으로 각 RGB채널에 대하여 연산을 해줘야 한다. 
    C_xflip[:,:,i] = np.matmul(A[:,:,i],B)            # 좌우반전
    C_yflip[:,:,i] = np.matmul(B,A[:,:,i])            # 상하반전
    C_xyflip[:,:,i] = np.matmul(C_yflip[:,:,i],B)     # 상하좌우반전

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,3,1)
a1.imshow(A)
a2 = fig.add_subplot(2,3,2)
a2.imshow(B)
a4 = fig.add_subplot(2,3,4)
a4.imshow(C_xflip)
a5 = fig.add_subplot(2,3,5)
a5.imshow(C_yflip)
a6 = fig.add_subplot(2,3,6)
a6.imshow(C_xyflip)
plt.show()