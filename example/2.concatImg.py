import numpy as np
import matplotlib.pyplot as plt

path = 'example/sampledata/lenna.png'
img = plt.imread(path)                                      # 원본영상
box = [128, 256, 128, 128]                                  # cropping하기위한 박스 위치 x, y, w, h
img_hconcat = np.hstack((img,)*2)
img_vconcat = np.vstack((img,)*2)

# plot
fig = plt.figure()
a1 = fig.add_subplot(2,2,1)    
a1.imshow(img)       # 원본 영상
a1.set_title('original image')
a2 = fig.add_subplot(2,2,3)
a2.imshow(img_hconcat)    # 마스크 영역
a2.set_title('horizon concat')
a3 = fig.add_subplot(2,2,4)
a3.imshow(img_vconcat)    # 마스크 영역
a3.set_title('vertical concat')
plt.show()
