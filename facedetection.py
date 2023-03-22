import cv2
import numpy as np
import os #for creating new folder and t check if a folder already exists 
cap=cv2.VideoCapture(0)
name=input("Enter your name:")
frames=[]
outputs=[] 
detector=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:   #we dont know till what point to capture the screen
    ret,frame=cap.read()
    if ret:   #ret has to be true so that we have somethig to reflect on the screen 
        faces=detector.detectMultiScale(frame)
        for face in faces:
            x,y,w,h=face
            cut=frame[y:y+h,x:x+w]
            #to fix the size of the window 
            fix=cv2.resize(cut,(100,100))
            gray=cv2.cvtColor(fix,cv2.COLOR_BGR2GRAY)
        cv2.imshow("My screen",frame)
        cv2.imshow("My face",gray)
    key=cv2.waitKey(1)   

    if key== ord("q"):
        break
    if key == ord("c"):
        #cv2.imwrite(name+".jpg", frame)
        frames.append(gray.flatten())
        outputs.append([name])

x=np.array(frames)
y=np.array(outputs)
data=np.hstack([y, x]) 
print(data.shape)
f_name="face_data.npy"
if os.path.exists(f_name):
    old=np.load(f_name)
    data=np.vstack([old,data])


np.save(f_name, data)

cap.release()
cap.destroyAllWindows()    

