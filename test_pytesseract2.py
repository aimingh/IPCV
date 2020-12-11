import pytesseract
from cv2 import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

def boxing(img, words):
    img_box = np.copy(img)
    n_boxes = len(words['text'])
    # Show with Debug Console
    for i in range(n_boxes):
        if int(words['conf'][i]) > 1:
            (x, y, w, h) = (words['left'][i], words['top'][i], words['width'][i], words['height'][i])
            img_box = cv2.rectangle(img_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # img_box = cv2.putText(img_box, words['text'][i], (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5,(0, 0, 255), 1)
            pill_image = Image.fromarray(img_box)
            # 나눔글꼴 설치
            # sudo apt-get install fonts-nanum*
            # 폰트캐쉬 삭제
            # sudo fc-cache -fv
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothicExtraBold.ttf'
            draw = ImageDraw.Draw(pill_image)
            draw.text((x, y), words['text'][i], font=ImageFont.truetype(font_path, 20), fill=(0, 0, 255))
            img_box = np.array(pill_image)
    return img_box

def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.resize(img, None,fx=2,fy=2, interpolation=cv2.INTER_AREA)
    kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) 
    output_1 = cv2.filter2D(img,-1,kernel_sharpen_1)

    kernel = np.ones((3,3),np.uint8)

    img_erosion = cv2.erode(output_1,kernel,iterations=1)
    img_dilation = cv2.dilate(img_erosion,kernel,iterations=1)

    bi_img = cv2.threshold(img_dilation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    final = np.hstack((img, output_1, img_erosion, img_dilation, bi_img))
    cv2.imshow('compare',final)
    cv2.waitKey()
    return bi_img

def preprocessing2(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img = cv2.resize(img, None,fx=2,fy=2, interpolation=cv2.INTER_AREA)

    kernel = np.ones((3,3),np.uint8)
    kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) 

    bi_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    img_erosion = cv2.erode(bi_img,kernel,iterations=1)
    img_dilation = cv2.dilate(img_erosion,kernel,iterations=1)

    output_1 = cv2.filter2D(img_dilation,-1,kernel_sharpen_1)

    final = np.hstack((img, bi_img, img_erosion, img_dilation, output_1))
    cv2.imshow('compare',final)
    cv2.waitKey()
    return bi_img

def get_OCR(filename):
    custom_config = r'--oem 3 --psm 6 -l kor+kor_vert+eng'

    img = cv2.imread(filename)

    bi_img = preprocessing2(img)
    msg = pytesseract.image_to_string(bi_img,config=custom_config)
    words = pytesseract.image_to_data(bi_img, config=custom_config,output_type=pytesseract.Output.DICT)
    print(msg)

    bi_img = cv2.cvtColor(bi_img, cv2.COLOR_GRAY2RGB)
    bi_img = boxing(bi_img, words)
    img = np.hstack((img, bi_img))
    cv2.imshow('img',img)
    cv2.waitKey()

if __name__ == "__main__":
    file_dir = './datas/receipt/crop'
    file_list = os.listdir(file_dir)
    for filename in file_list:
        get_OCR(file_dir + '/' + filename)

