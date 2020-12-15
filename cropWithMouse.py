from cv2 import cv2
import numpy as np

def mouse_event():
    global ix, iy, drawing, result,img
    drawing = False # true if mouse is pressed
    ix, iy = -1, -1
    img = np.zeros((512, 512, 3), np.uint8)
    result = np.zeros((512, 512, 3), np.uint8)

    def draw_shape(event, x, y, flags, param): # mouse callback function
        global ix, iy, drawing, result,img # Need to define global
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True 
            ix, iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True: 
                result = img.copy()
                cv2.rectangle(result, (ix, iy), (x, y), (0, 255, 0), 0)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
    cv2.namedWindow('image'); cv2.setMouseCallback('image', draw_shape)
    while(1):
        if drawing ==True:
            cv2.imshow('image', result)
        else:
            cv2.imshow('image', img)
        if cv2.waitKey(1) == ord('q'):
            break 

if __name__ == "__main__":
    mouse_event()