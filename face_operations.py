import cv2
import os
from PIL import Image
import numpy as np
from data_operations import *

def add_data():
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video width
    cam.set(4, 480) # set video height

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # For each person, enter one numeric face id

    value=input("\nName of the criminal:\t")
    id=int(input("Case ID:\t"))
    inf=input("Crime details:\t")


    print("\n\t\t [INFO] Initializing face capture. Look the camera and wait ...")
    # Initialize individual sampling face count
    count = 0

    while(True):

        ret, img = cam.read()
        # img = cv2.flip(img, -1) # flip video image vertically
        #gray = cv2.cvtColor(img, cv2.COLOR_GRAY)
        faces = face_detector.detectMultiScale(img, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User."+ str(id) + '.' + str(count) + ".jpg",img[y:y+h,x:x+w])

            cv2.imshow('Capturing the face', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30: # Take 30 face sample and stop video
             break

    # Do a bit of cleanup
    print("\n\t\t [INFO] Exiting Program and cleanup stuff \n")
    cam.release()
    cv2.destroyAllWindows()
    add(value,id,inf)

def train_model():
    # Path for face image database
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    # function to get the images and label data
    try:
        def getImagesAndLabels(path):

            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []

            for imagePath in imagePaths:

                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')

                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = detector.detectMultiScale(img_numpy)

                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
                

            return faceSamples,ids

    except:
        print("No images to train the model . ")

    print ("\n\t\t [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n\t\t [INFO] {0} faces trained. Exiting Program\n".format(len(np.unique(ids))),"\n")


def recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_PLAIN

    #iniciate id counter
    l=data()
    

    id=0
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640) # set video widht
    cam.set(4, 480) # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, img =cam.read()
        # img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
           )

        for(x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            global n
            n=0

            for i in range(1,len(l)):
                if l[i]['Case ID']==id:
                    n=i
                    break

            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 100):
                id = l[n]
                
                confidence = "  {0}%".format(round(100 - confidence))
                
            else:
                id = "Unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            if id!="Unknown" and id!="None":
                    print(id)
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('Identify the criminal',img)

        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n\t\t [INFO] Exiting Program and cleanup stuff\n")
    cam.release()
    cv2.destroyAllWindows()
