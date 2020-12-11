import pytesseract
from cv2 import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

def rgb2binary(img, threshold=160):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bi_img = np.zeros(img.shape, np.uint8)
    bi_img[np.where(img>threshold)] = 255
    return bi_img

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
            font_path = '/usr/share/fonts/truetype/nanum/NanumGothicExtraBold.ttf'
            draw = ImageDraw.Draw(pill_image)
            draw.text((x, y), words['text'][i], font=ImageFont.truetype(font_path, 20), fill=(0, 0, 255))
            img_box = np.array(pill_image)
    return img_box

def get_OCR(filename):
    custom_config = r'--oem 3 --psm 6 -l kor+kor_vert+eng'

    img = cv2.imread(filename)
    bi_img = rgb2binary(img)
    bi_img2 = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    msg = pytesseract.image_to_string(bi_img,config=custom_config)
    words1 = pytesseract.image_to_data(img, config=custom_config,output_type=pytesseract.Output.DICT)
    words2 = pytesseract.image_to_data(bi_img, config=custom_config,output_type=pytesseract.Output.DICT)
    words3 = pytesseract.image_to_data(bi_img2, config=custom_config,output_type=pytesseract.Output.DICT)
    print(msg)

    bi_img = cv2.cvtColor(bi_img, cv2.COLOR_GRAY2RGB)
    bi_img2 = cv2.cvtColor(bi_img2, cv2.COLOR_GRAY2RGB)
    img_box1 = boxing(img, words1)
    img_box2 = boxing(bi_img, words2)
    img_box3 = boxing(bi_img2, words3)

    # img1 = np.hstack((img, bi_img, bi_img2))
    img = np.hstack((img_box1, img_box2, img_box3))
    # img = np.vstack((img1, img2))
    cv2.imshow('img',img)
    cv2.waitKey()

if __name__ == "__main__":
    file_dir = './datas/receipt/crop'
    file_list = os.listdir(file_dir)
    for filename in file_list:
        get_OCR(file_dir + '/' + filename)

