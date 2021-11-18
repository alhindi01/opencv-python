import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() 
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x,y,w,h)
        #region of interest
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #built in opencv recognizer
        #no deep learning libraries 
        id_, conf = recognizer.predict(roi_gray)
        if conf>= 45 and conf <= 85:
            # draw rectangle
            rectangle_color = (0, 0, 255)#BGR(blue green red) NOT RPG
            stroke = 3
            end_core_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x,y), (end_core_x, end_cord_y), rectangle_color, stroke)

            
            # print(id_)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            criminal =  "criminal : "  + name

            color = (255, 255, 255)
            criminal_color = (0,0,255)
            stroke = 3
            cv2.putText(frame, criminal, (x,y), font, 1, criminal_color, stroke, cv2.LINE_AA)
        else:
            # draw rectangle
            rectangle_color = (0, 255, 0)#BGR(blue green red) NOT RPG
            stroke = 3
            end_core_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x,y), (end_core_x, end_cord_y), rectangle_color, stroke)
            cv2.putText(frame, "citizen", (x,y), font, 1, color, stroke, cv2.LINE_AA)




        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        # draw rectangle
        # color = (255, 0, 0)#BGR(blue green red) NOT RPG
        # stroke = 3
        # end_core_x = x + w
        # end_cord_y = y + h
        # cv2.rectangle(frame, (x,y), (end_core_x, end_cord_y), color, stroke)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


