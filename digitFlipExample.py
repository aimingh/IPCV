import numpy as np
import matplotlib.pyplot as plt

A = np.array([[0,0,0,0,0,0,0,0,0,0], # 원본영상, 숫자2
            [0,0,1,1,1,1,1,1,0,0],
            [0,1,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,1,1,0],
            [0,0,1,1,1,1,1,1,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,0],
            [0,0,0,0,0,0,0,0,0,0],])
B = np.identity(n=10)[::-1]         # 입력 행렬 B

C_xflip = np.matmul(A,B)            # 좌우반전  AB
C_yflip = np.matmul(B,A)            # 상하반전  BA
C_xyflip = np.matmul(C_yflip,B)     # 상하좌우반전  BAB

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,3,1)
a1.imshow(A, cmap='gray')
a2 = fig.add_subplot(2,3,2)
a2.imshow(B, cmap='gray')
a4 = fig.add_subplot(2,3,4)
a4.imshow(C_xflip, cmap='gray')
a5 = fig.add_subplot(2,3,5)
a5.imshow(C_yflip, cmap='gray')
a6 = fig.add_subplot(2,3,6)
a6.imshow(C_xyflip, cmap='gray')
plt.show()