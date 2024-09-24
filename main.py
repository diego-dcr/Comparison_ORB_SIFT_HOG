from importlib.resources import path
from sre_constants import SUCCESS
from venv import main
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import imutils
from imutils.object_detection import non_max_suppression


class ORB:

    orb = cv2.ORB_create(nfeatures = 1000)
    imgTrain = []
    imgClassNames = []
    desList = []
    id = -1

    def __init__(self, path):
        myList = os.listdir(path)
        for cl in myList:
            imgCur = cv2.imread(f'{path}/{cl}', 0)
            self.imgTrain.append(imgCur)
            self.imgClassNames.append(os.path.splitext(cl)[0])

    def findQueryDes(self):
        for img in (self.imgTrain):
            kp, des = self.orb.detectAndCompute(img, None)
            self.desList.append(des)

    def findID(self, img):
        self.img = img.copy()
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.orb.detectAndCompute(img, None)
        bf = cv2.BFMatcher()
        matchList = []
        try:
            for qd in self.desList:
                matches = bf.knnMatch(qd, des, k = 2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                matchList.append(len(good))
        except:
            pass

        if len(matchList) != 0:
            if max(matchList) > 10: #Cuidado con la constante (10) de los matches points
                self.id = matchList.index(max(matchList))

    def nameImg(self):
        if self.id != -1:
            cv2.putText(self.img, self.imgClassNames[self.id], (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 4)

    def showImg(self):
        self.img = cv2.resize(self.img, (0,0), fx = 0.3, fy = 0.3)
        cv2.imshow("Resultado", self.img)
        cv2.waitKey(0)

class SIFT:

    sift = cv2.xfeatures2d.SIFT_create()
    imgTrain = []
    imgClassNames = []
    desList = []
    id = -1

    def __init__(self, path):
        myList = os.listdir(path)
        for cl in myList:
            imgCur = cv2.imread(f'{path}/{cl}', 0)
            self.imgTrain.append(imgCur)
            self.imgClassNames.append(os.path.splitext(cl)[0])

    def findQueryDes(self):
        for img in (self.imgTrain):
            kp, des = self.sift.detectAndCompute(img, None)
            self.desList.append(des)

    def findID(self, img):
        self.img = img.copy()
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(img, None)
        bf = cv2.BFMatcher()
        matchList = []
        try:
            for qd in self.desList:
                matches = bf.knnMatch(qd, des, k = 2)
                good = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                matchList.append(len(good))
        except:
            pass

        if len(matchList) != 0:
            if max(matchList) > 10: #Cuidado con la constante (10) de los matches points
                self.id = matchList.index(max(matchList))

    def nameImg(self):
        if self.id != -1:
            cv2.putText(self.img, self.imgClassNames[self.id], (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 4)

    def showImg(self):
        self.img = cv2.resize(self.img, (0,0), fx = 0.3, fy = 0.3)
        cv2.imshow("Resultado", self.img)
        cv2.waitKey(0)

class HOG:

    hog = cv2.HOGDescriptor()
    bounding_boxes = []
    # img = imagen a la que se le aplicará el filtro

    def __init__(self, img):
        self.img = img
        scale = 1.0
        w = int(self.img.shape[1] / scale)
        self.img = imutils.resize(self.img, width = w)
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def findPeople(self):
        (self.bounding_boxes, weights) = self.hog.detectMultiScale(self.img, winStride = (4, 4), 
                                                                   padding = (8, 8), scale = 1.05)
    
    def drawPeople(self):
        self.bounding_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in self.bounding_boxes])
        pick = non_max_suppression(self.bounding_boxes, probs = None, overlapThresh = 0.65)
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(self.img, (xA, yA), (xB, yB), (0, 255, 0), 2)

    def showImg(self):
        cv2.imshow("Resultado", self.img)
        cv2.waitKey(0)

def main():
    path = "imgTrain"

    # Para ORB y SIFT
    # img = "imgQuery/Alebrije_2.jpg"
    # img = "imgQuery/El_Libro_Salvaje_2.jpg"
    img = "imgQuery/Libro_Infantil_2.jpg"

    # Para HOG
    # img = "imgTrain/Personas.jpg"
    # img = "imgTrain/PersonasCaminando.jpg"
    # img = "imgTrain/PersonasCaminando2.jpg"

    # ORB Descriptor

    orbDescriptor = ORB(path)
    orbDescriptor.findQueryDes()
    orbDescriptor.findID(cv2.imread(img))
    orbDescriptor.nameImg()
    orbDescriptor.showImg()

    # SIFT Descriptor
    # Falla con la imagen del alebrije
    
    # siftDescriptor = SIFT(path)
    # siftDescriptor.findQueryDes()
    # siftDescriptor.findID(cv2.imread(img))
    # siftDescriptor.nameImg()
    # siftDescriptor.showImg()

    # HOG Descriptor
    # hogDescriptor = HOG(cv2.imread(img))
    # hogDescriptor.findPeople()
    # hogDescriptor.drawPeople()
    # hogDescriptor.showImg()

if __name__ == "__main__":
    main()