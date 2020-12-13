import numpy as np
import matplotlib.pyplot as plt

def main():
    path = 'example/sampledata/lenna.png'
    img = plt.imread(path)                                      # 원본영상
    box = [128, 256, 128, 128]                                  # cropping하기위한 박스 위치 x, y, w, h
    img_crop = img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]  # cropping

    # plot
    fig = plt.figure()
    a1 = fig.add_subplot(1,2,1)    
    a1.imshow(img)       # 원본 영상
    a1.set_title('original image')
    a2 = fig.add_subplot(1,2,2)
    a2.imshow(img_crop)    # 마스크 영역
    a2.set_title('cropping image')
    plt.show()

if __name__ == "__main__":
    main()