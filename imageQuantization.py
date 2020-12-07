import numpy as np
import matplotlib.pyplot as plt

depth_level = 32
# rgb2gray='gray'
rgb2gray='rgb'

path = 'lenna.png'
A = plt.imread(path)            # 원본영상

if rgb2gray=='gray':    # RGB -> GRAY
    A = np.sqrt(np.power(A[:,:,0],2) + np.power(A[:,:,1],2) + np.power(A[:,:,2],2))
else:                  
    rgb2gray=None
    
A_ = np.round((depth_level-1)*A)/(depth_level-1)    # N level로 영상을 바꾼다.

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,3,1)
a1.imshow(A, cmap=rgb2gray)
a2 = fig.add_subplot(2,3,2)
a2.imshow(A_, cmap=rgb2gray)
plt.show()