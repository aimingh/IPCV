import os
import cv2.data as cvdata
from cv2 import cv2 
import numpy as np

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

if __name__ == "__main__":
    train()