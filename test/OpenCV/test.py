import cv2 

cap = cv2.VideoCapture(0) 

face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml") 
eye_detect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")     

while True:
    _, frame = cap.read()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_detect.detectMultiScale(gray, 1.5, 5)  

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y), (x+w, y+h), (255,0,0), 5)  
        roi_gray = gray[y:y+h, x:x+h] 
        roi_color = frame[y:y+h, x:x+w] 

        eyes = eye_detect.detectMultiScale(roi_gray,1.5,5) 

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+ey), (0,255,0), 5)    

    cv2.imshow("Capture", frame)  
    if cv2.waitKey(1) == ord('q'):
        break 
 