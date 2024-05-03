from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import os

face_classifier = cv2.CascadeClassifier(r'C:\Users\deshr\OneDrive\Desktop\Facial Recognition Kaggle\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\deshr\OneDrive\Desktop\Facial Recognition Kaggle\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)



faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

size = 4
haar_file = (cv2.data.haarcascades + "haarcascade_frontalface_default.xml") #Note the change
datasets = r'C:\Users\deshr\OneDrive\Desktop\Facial Recognition Kaggle\datasets'


print('Recognizing Face Please Be in sufficient Lights...')

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
	for subdir in dirs:
		names[id] = subdir
		subjectpath = os.path.join(datasets, subdir)
		for filename in os.listdir(subjectpath):
			path = subjectpath + '/' + filename
			label = id
			images.append(cv2.imread(path, 0))
			labels.append(int(label))
		id += 1
(width, height) = (130, 100)

# Create a Numpy array from the two lists above
(images, labels) = [np.array(lis) for lis in [images, labels]]

# OpenCV trains a model from the images

model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)





while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        prediction2 = model.predict(roi_gray)		
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)


        if prediction2[1]<80:
            cv2.putText(frame, names[prediction2[0]], (x+40, y+40),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        else:
            cv2.putText(frame, 'not recognized',(x+40, y+40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()