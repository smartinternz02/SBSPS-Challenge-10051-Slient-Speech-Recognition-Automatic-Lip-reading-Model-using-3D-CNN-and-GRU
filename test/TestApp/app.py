import cv2 
import streamlit as st 
from PIL import Image 
import mediapipe as mp 


st.set_page_config(page_title = "Deep Learning Webapp", layout= "wide")  

st.markdown("""
    <style>
        *{
            text-align:center; 
        }
    </style> 
""", unsafe_allow_html=True)  

st.title("Streamlit Based Face Mesh App") 

col1, col2, col3 = st.columns(3)  

col1.markdown("## Built with Intelligence") 

c = st.container() 

FRAME = c.image([], use_column_width=True)   

st.sidebar.title("Mesh Settings") 

#Media Pipe 

mpFaceMesh = mp.solutions.face_mesh 
mpDraw = mp.solutions.drawing_utils

max_num_faces = st.sidebar.number_input("Maximum number of Faces: ", min_value = 1, max_value = 5) 

thickness = st.sidebar.slider("Mesh Thickness", 1,10,1)  

circle_radius = st.sidebar.slider("Circle Radius", 1,5,1)   

face_mesh = mpFaceMesh.FaceMesh(max_num_faces = max_num_faces)   

drawingSpec = mpDraw.DrawingSpec(thickness = thickness, circle_radius = circle_radius)    

#openCV

cap = cv2.VideoCapture(0)   

while True:
    status, img = cap.read() 
    if(status):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        results = face_mesh.process(imgRGB)     

        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                mpDraw.draw_landmarks(imgRGB, face_landmark, mpFaceMesh.FACEMESH_CONTOURS)  
                print(len(face_landmark.landmark))     


        FRAME.image(imgRGB)  
    else :
        break 

    if(cv2.waitKey(1) == ord('q')):
        break 

cap.release() 
cv2.destroyAllWindows()  



LIPS_UPPER_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291] # ( left to right ) 

LIPS_LOWER_OUTER = [375, 321, 405, 314, 17, 84, 181, 91, 146, 61] # in reverse Order to complete the shell of Lips (right to left) 