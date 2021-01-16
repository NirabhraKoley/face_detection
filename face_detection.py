import cv2 as cv

img = cv.imread('F:\\opencv\\images\\people3.webp')
# cv.imshow('People', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('frontface_detection.xml')

face_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

print(f'Number Of Faces Found = {len(face_rect)}')

for (x, y, w, h) in face_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
cv.imshow('Detected', img)

cv.waitKey(0)
