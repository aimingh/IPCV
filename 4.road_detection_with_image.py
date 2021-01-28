import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

# def preprocessing(img):
#     gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (3,3),0)
#     gray = cv2.equalizeHist(gray)
#     return gray

# def preprocessing2(img):
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     gray = cv2.GaussianBlur(img_hsv[:,:,1], (3,3),0)
#     gray = cv2.equalizeHist(gray)
#     return gray

# def watershed(img_gray):
#     mean = np.average(img_gray)
#     _, thresh1 = cv2.threshold(img_gray,mean,255,cv2.THRESH_BINARY_INV)
#     _, thresh2 = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)
#     thresh = np.bitwise_or(thresh1, thresh2)

#     kernel = np.ones((3,3),np.uint8)
#     opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

#     sure_bg = cv2.dilate(opening,kernel,iterations=2)

#     dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)
#     _, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
#     sure_fg = np.uint8(sure_fg)

#     unknown = cv2.subtract(sure_bg, sure_fg)

#     ret, markers = cv2.connectedComponents(sure_fg)
#     markers = markers + 1
#     markers[unknown == 255] = 0

#     markers = cv2.watershed(img,markers)
#     return markers

# def drawContours(img_rgb, contours):
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         print(f'area of contours: {area}')
#         cv2.drawContours(img_rgb, [cnt], 0, (255,0,0), 1)
#     # cv.drawContours(img_rgb, contours, 0, (255,0,0), 2)
#     return img_rgb

# def approximationContour(img, contours, e=0.02):
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         print(f'x: {x}, y: {y}, w: {w}, h: {h}')
#         epsilon = e*cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)
#         print(len(approx))
#         cv2.drawContours(img, [approx], 0, (0,255,255), 2)
#     return img

# def rectwithname(img, contours, e=0.02):
#     result = img.copy()
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         epsilon = e*cv2.arcLength(cnt, True)
#         approx = cv2.approxPolyDP(cnt, epsilon, True)

#         if len(approx)>3 and w<30 and h<50:
#             cv2.rectangle(result,(x,y),(x+w,y+h),(255,0,255),2)
#     return result

# def segment_road(img):
#     gray = preprocessing(img)
#     markers = watershed(gray)

#     gray[np.where(markers>1)] = 255
#     gray[:int(0.5*gray.shape[0]),:] = 255
#     _, img_bi = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
#     closing = cv2.morphologyEx(img_bi,cv2.MORPH_CLOSE, np.ones((3,3),np.uint8),iterations=2)
#     return closing
# def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
#     # If there are no lines to draw, exit.
#     if lines is None:
#         return
#     # Make a copy of the original image.
#     img = np.copy(img)
#     # Create a blank image that matches the original in size.
#     line_image = np.zeros(
#         (
#             img.shape[0],
#             img.shape[1],
#             3
#         ),
#         dtype=np.uint8,
#     )
#     # Loop over all lines and draw them on the blank image.
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
#     # Merge the image with the lines onto the original.
#     img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
#     # Return the modified image.
#     return img

def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7,7),0)
    return gray

def segment_road(img):
#     gray = preprocessing(img)
#     markers = watershed(img, gray)

#     gray[np.where(markers>1)] = 255
#     gray[:int(0.5*gray.shape[0])+ 0,:] = 255
#     _, img_bi = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    closing = cv2.morphologyEx(gray,cv2.MORPH_CLOSE, np.ones((3,3),np.uint8),iterations=2)
    return closing


def preprocessing(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (7,7),0)
    return gray


def watershed(img, img_gray):
#     mean = np.average(img_gray)
#     _, thresh1 = cv2.threshold(img_gray,mean,255,cv2.THRESH_BINARY_INV)
#     _, thresh2 = cv2.threshold(img_gray,200,255,cv2.THRESH_BINARY)
#     thresh = np.bitwise_or(thresh1, thresh2)
    _, thresh = cv2.threshold(img_gray,np.average(img_gray)-40,255,cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)

    sure_bg = cv2.dilate(opening,kernel,iterations=2)

    dist_transform = cv2.distanceTransform(sure_bg,cv2.DIST_L2,5)
    _, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    unknown = cv2.subtract(sure_fg, sure_bg)

    ret, markers = cv2.connectedComponents(unknown)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img,markers)
    return dist_transform


filenames = ['snapshot/b8f3506c-3f65-11eb-9798-16f63a1aa8c9.jpg',
            'snapshot/cad5dade-3f65-11eb-9798-16f63a1aa8c9.jpg',
            'snapshot/d9790c50-3f65-11eb-9798-16f63a1aa8c9.jpg',
            'snapshot/e74fce5e-3f65-11eb-9798-16f63a1aa8c9.jpg']

imgs = []
titles = []
for i, filename in enumerate(filenames):
    img = plt.imread(filename)

    # # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # gray = cv2.GaussianBlur(img_hsv[:,:,2], (3,3),0)
    # gray = cv2.equalizeHist(gray)
    # _, thresh = cv2.threshold(gray,60,255,cv2.THRESH_BINARY_INV)
    # img_th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,4)


    # img_bi = segment_road(img)
    # imcanny = cv2.Canny(img_bi, 50, 200)
    # # contours, _ = cv2.findContours(img_bi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # # img_rgb = drawContours(img, contours)
    # # img_approx = rectwithname(img, contours)

    # lines = cv2.HoughLinesP(
    #     imcanny,
    #     rho=6,
    #     theta=np.pi / 60,
    #     threshold=160,
    #     lines=np.array([]),
    #     minLineLength=40,
    #     maxLineGap=25
    # )

    # line_image = draw_lines(img, lines)

    img_gray = preprocessing(img)
    img_water = watershed(img, img_gray)

    imgs.append(img_gray)
    imgs.append(img_water)
    titles.append(f"img_{i}")
    titles.append(f"result_{i}")


for i in range(len(imgs)):
    plt.subplot(2,4,i+1),plt.imshow(imgs[i]),plt.title(''),plt.xticks([]),plt.yticks([])
plt.show()