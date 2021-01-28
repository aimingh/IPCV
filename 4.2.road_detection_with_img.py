import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7,7),0)
    return gray

def thresholding(img_gray):
    _, img_th = cv2.threshold(img_gray,np.average(img_gray)-32,255,cv2.THRESH_BINARY)
    img_th2 = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,7)
    img_th3 = np.bitwise_and(img_th, img_th2)
    img_th4 = cv2.subtract(img_th2, img_th3)
    for i in range(5):
        img_th4 = cv2.medianBlur(img_th4, 5)
    return img_th4

def mask_roi(img_th, roi):
    mask = np.zeros_like(img_th)
    cv2.fillPoly(mask, np.array([roi], np.int32), 255)
    masked_image = cv2.bitwise_and(img_th, mask)
    return masked_image

def drawContours(img_rgb, contours):
    for cnt in contours:
        area = cv2.contourArea(cnt)
        cv2.drawContours(img_rgb, [cnt], 0, (255,0,0), 1)
    return img_rgb

def approximationContour(img, contours, e=0.02):
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        epsilon = e*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(img, [approx], 0, (0,255,255), 2)
    return img

def rectwithname(img, contours, e=0.02):
    result = img.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        epsilon = e*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

#         if len(approx)<7:
        cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,255),2)
    return result

filenames = ['snapshot/b8f3506c-3f65-11eb-9798-16f63a1aa8c9.jpg',
            'snapshot/cad5dade-3f65-11eb-9798-16f63a1aa8c9.jpg',
            'snapshot/d9790c50-3f65-11eb-9798-16f63a1aa8c9.jpg',
            'snapshot/e74fce5e-3f65-11eb-9798-16f63a1aa8c9.jpg']

imgs = []
titles = []
width = 224
height = 224
roi = [(0, height),(0, height/2-30), (width, height/2-30),(width, height),]
for i, filename in enumerate(filenames):
    img = plt.imread(filename)

    img_gray = preprocessing(img)
    img_th = thresholding(img_gray)
    img_roi = mask_roi(img_th, roi)
    
    kernel = np.ones((5,3),np.uint8)
    img_cl = cv2.morphologyEx(img_roi,cv2.MORPH_CLOSE, np.ones((5,5),np.uint8),iterations=5)
    img_op = cv2.morphologyEx(img_cl,cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
    
    cannyed_image = cv2.Canny(img_op, 300, 500)
    contours, _ = cv2.findContours(cannyed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    img_rgb = drawContours(img, contours)
    img_approx = approximationContour(img, contours, e=0.01)   
    img_approx_rect = rectwithname(img, contours, e=0.01)   


    center_image_point = [height-1, width/2-1]
    center_ptrs = []
    e=0.01
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        center_ptr = [y + 0.5*h , x + 0.5*w,]

        # epsilon = e*cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        # x, _, y = approx.shape
        # center_ptr = np.average(approx.reshape((x,y)),0)

        center_ptrs.append(center_ptr)
    center_ptrs = np.array(center_ptrs)

    L2_norm = np.linalg.norm((center_ptrs - center_image_point), axis=1, ord=2)
    loc = np.where(L2_norm==L2_norm.min())[0][0]

    midlane = center_ptrs[loc]

    cv2.line(img_gray,(int(midlane[1]),int(midlane[0])),(int(center_image_point[1]), int(center_image_point[0])),(255,255,255),3)

    imgs.append(img_gray)
    imgs.append(img_approx_rect)
    titles.append(f"img_{i}")
    titles.append(f"result_{i}")


for i in range(len(imgs)):
    plt.subplot(2,4,i+1),plt.imshow(imgs[i]),plt.title(''),plt.xticks([]),plt.yticks([])
plt.show()



# def segment_road(img):
# #     gray = preprocessing(img)
# #     markers = watershed(img, gray)

# #     gray[np.where(markers>1)] = 255
# #     gray[:int(0.5*gray.shape[0])+ 0,:] = 255
# #     _, img_bi = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
#     closing = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, np.ones((3,3),np.uint8),iterations=2)
#     return closing

# def watershed(img, img_gray):
# #     mean = np.average(img_gray)
# #     _, thresh1 = cv2.threshold(img_gray,mean,255,cv2.THRESH_BINARY_INV)
# #     _, thresh2 = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)
# #     thresh = np.bitwise_or(thresh1, thresh2)
#     _, thresh = cv2.threshold(img_gray,np.average(img_gray)-20,255,cv2.THRESH_BINARY)

#     kernel = np.ones((3,3),np.uint8)
#     opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

#     sure_bg = cv2.dilate(opening,kernel,iterations=2)

#     dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)
#     _, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
#     sure_fg = np.uint8(sure_fg)

#     unknown = cv2.subtract(sure_bg, sure_fg)

#     ret, markers = cv2.connectedComponents(sure_bg)
#     markers = markers + 1
# #     markers[unknown == 255] = 0

#     markers = cv2.watershed(img,markers)
#     return markers