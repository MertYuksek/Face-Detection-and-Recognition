import cv2 as ocv

img = ocv.imread("Photos/group 1.jpg")
#ocv.imshow("Lady" ,img)

# Firstly we should convert img to grayscale, 
# because face detection does not involve colors, just be intereted in objets,
# using edges it determines is there any face.
gray_img = ocv.cvtColor(img, ocv.COLOR_BGR2GRAY)
#ocv.imshow("Gray Lady" ,gray_img)

# We get classifier
haar_cascade = ocv.CascadeClassifier("haar_face.xml")

# We get rectangle list which include faces in the img.
# Haar Cascade sensitive the noise. We can change scaleFactor and minNeighbor 
# to decrease sensitivity.
face_rects = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1)

for (x,y,w,h) in face_rects:
    ocv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

ocv.imshow("Detected Faces", img)

print(len(face_rects))

ocv.waitKey(0)



