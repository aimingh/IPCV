import os
import cv2.data as cvdata
from cv2 import cv2 

def face_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces is not None:
        for (x,y,w,h) in faces: 
            return frame[y:y+h, x:x+w, :]
    else:
        return None

dir_name = os.getcwd() + '/datas/images/faces/'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

face_cascade = cv2.CascadeClassifier(cvdata.haarcascades + 'haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)

count = 0

try:
    print(f'start, capture.isOpened(): {capture.isOpened()}')
    while capture.isOpened():
        ret, frame = capture.read()
        put_text = ''

        cropped_area = face_detector(frame)

        if cropped_area is not None:
            count += 1
            area = cv2.resize(cropped_area, (200,200))
            area = cv2.cvtColor(area, cv2.COLOR_RGB2GRAY)

            file_name = dir_name + 'user' + str(count) + '.jpg'
            cv2.imwrite(file_name, area)
            put_text = f"Face Found!, imwrite() count: {count}"
            cv2.putText(area, put_text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropped', area)
        else:
            put_text = f"Face not Found, imwrite() count: {count}"

        cv2.putText(frame, put_text, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1)==ord('q') or count==1000:
            break
except:
    print('Error')
finally:
    print('finished')
    capture.release()
    cv2.destroyAllWindows()