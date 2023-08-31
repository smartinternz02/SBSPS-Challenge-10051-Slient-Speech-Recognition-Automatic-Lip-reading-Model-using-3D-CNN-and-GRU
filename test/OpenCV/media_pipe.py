import cv2 
import mediapipe as mp 
import time 

cap = cv2.VideoCapture(0) 

pTime = 0 

mpDraw = mp.solutions.drawing_utils 
mpFaceMesh = mp.solutions.face_mesh 
face_mesh = mpFaceMesh.FaceMesh(max_num_faces = 2) 

drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)  

while True:
    status, frame = cap.read()
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    results = face_mesh.process(imgRGB) 
    if results.multi_face_landmarks:
        for faceLMs in results.multi_face_landmarks: 
            mpDraw.draw_landmarks(frame, faceLMs, mpFaceMesh.FACEMESH_CONTOURS, drawSpec)    


    cTime = time.time() 
    fps = 1/(cTime - pTime) 
    cv2.putText(frame,f"FPS:{int(fps)}",(20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)        
    pTime = cTime 
    cv2.imshow("Image",frame) 
    if cv2.waitKey(1) == ord('q'): 
        break 

cap.release() 
cv2.destroyAllWindows() 