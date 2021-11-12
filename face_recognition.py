import cv2 as ocv
import numpy as npy

haar_cascade = ocv.CascadeClassifier("haar_face.xml")

people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']

face_recognizer = ocv.face.LBPHFaceRecognizer_create()  
# Get trained rocognizer.
face_recognizer.read("trained_face_recognition.yml")

img = ocv.imread("FacesImg/val/ben_afflek/2.jpg")
#ocv.imshow("Img", img)

gray = ocv.cvtColor(img, ocv.COLOR_BGR2GRAY)
face_rects = haar_cascade.detectMultiScale(gray,
             scaleFactor=1.1, minNeighbors=4)

for (x,y,w,h) in face_rects:
    face = gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(face)
    print(f'Label = {people[label]} with a confidence of {confidence}')

    ocv.putText(img, str(people[label]), (20,20), ocv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    ocv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

ocv.imshow('Detected Face', img)

ocv.waitKey(0)