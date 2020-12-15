import os
import cv2.data as cvdata
from cv2 import cv2 
import numpy as np

def face_detector(frame):
    face_cascade = cv2.CascadeClassifier(cvdata.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if faces is not None:
        for (x,y,w,h) in faces: 
            return frame[y:y+h, x:x+w, :]
    else:
        return None
        
def train():
    dir_name = os.getcwd() + '/datas/images/faces/'
    if not os.path.exists(dir_name):
        return 0

    onlyfiles = [f for f in os.listdir(dir_name) if os.path.isfile(os.path.join(dir_name,f))]

    Training_Data, Labels = list(), list()
    for i, files in enumerate(onlyfiles):
        image_path = dir_name + files
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    model.save("tmp.yml")

   

def test():
    model = cv2.face.LBPHFaceRecognizer_create()
    model.read("tmp.yml")
    capture = cv2.VideoCapture(0)
    while capture.isOpened():
        ret, frame = capture.read()
        cascade_face = face_detector(frame)
        try:
            face = cv2.cvtColor(cascade_face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            display_string = ''
            if result[1] < 500:
                confidence = int(100*(1-(result[1]/300)))
                display_string = f'{confidence}% Confidence, '
            if confidence > 75:
                display_string = display_string + "Unlocked"
            else:
                display_string = display_string + "Locked"
            cv2.putText(frame, display_string, (250,450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        except:
            print("not Found")
        cv2.imshow("frame", frame)
        if cv2.waitKey(1)==ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # train()
    test()