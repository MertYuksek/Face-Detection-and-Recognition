import os
import cv2 as ocv
import numpy as npy

haar_cascade = ocv.CascadeClassifier("haar_face.xml")

# Photos folder of people
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
DIR = "FacesImg/train"

# Data contains all people faces.
data = []
# Labels contains person names
labels = []

# Here we fill arrays which are above.
def create_train():
    for person in people:
        path = os.path.join(DIR,person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_array = ocv.imread(img_path)
            gray = ocv.cvtColor(img_array, ocv.COLOR_BGR2GRAY)
            face_rects = haar_cascade.detectMultiScale(gray,
             scaleFactor=1.1, minNeighbors=4)
            
            for (x,y,w,h) in face_rects:
                face = gray[y:y+h, x:x+w]
                data.append(face)
                labels.append(label)

create_train()

data = npy.array(data, dtype="object")
labels = npy.array(labels)

# Implemented face recognizer in OpneCV
face_recognizer = ocv.face.LBPHFaceRecognizer_create()               

# Train the recognizer
face_recognizer.train(data,labels)

# Save face recognizer
face_recognizer.save("trained_face_recognition.yml")   
npy.save('data.npy', data)
npy.save('labels.npy', labels)



print(f'Length of data = {len(data)}')
print(f'Length of labels = {len(labels)}')
