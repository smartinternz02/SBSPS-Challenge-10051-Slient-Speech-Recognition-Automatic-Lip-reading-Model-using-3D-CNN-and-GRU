#import mediapipe as mp 
import numpy as np 
import cv2 
import os 
import dlib 
from typing import List 
import tensorflow as tf 


# os.environ["TF_CPP_MIN_LOG_LEVEL"] = 3 
try:
    gpu = tf.config.experimental.list_physical_devices("GPU")[0] 
    tf.config.experimental.set_memory_growth(gpu,True) 
except:
    pass 

video_path = "test_two.mp4" 




#print(mpFaceMesh.FACEMESH_LIPS)  

# for keys in mpFaceMesh.__dict__: 
#     if "FACEMESH_" in keys:
#         print(keys)  


#  Detecting Face Mesh, Disabiling as of now 

# def landmark_facemesh(video_path:str):  
#     mpFaceMesh = mp.solutions.face_mesh 
#     face_mesh = mpFaceMesh.FaceMesh(max_num_faces = 1) 
#     mpDraw = mp.solutions.drawing_utils 
#     drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2) 
#     cap = cv2.VideoCapture(video_path) 
#     frames = [] 
#     # for _ in range(int(cap.get(7))):
#     while True: 
#         status, frame = cap.read() 
#         if not status:
#             break
#         imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  

#         results = face_mesh.process(imgRGB)  

#         if(results.multi_face_landmarks): 
#             landmark = results.multi_face_landmarks[0] 
#             mpDraw.draw_landmarks(imgRGB, landmark, mpFaceMesh.FACEMESH_LIPS, drawSpec)    
            
#             frames.append(imgRGB)  
#         #cv2.imshow("Frame", frame)  

#         # if(cv2.waitKey(1) == ord('q')):
#             # break   

#     # cap.release() 
#     # cv2.destroyAllWindows()

#     return frames 


def load_video(video_path, display = False, output_path = False):   

    cap = cv2.VideoCapture(video_path)  

    face_detector = dlib.get_frontal_face_detector() 

    CROP_HEIGHT = 46
    CROP_WIDTH = 140

    FACE_HEIGHT = 0.5 

    lip_frames = [] 

    target_width = target_height = None 

    if output_path: 
        output_fourcc = cv2.VideoWriter_fourcc(*"MP4V")      
        output = cv2.VideoWriter(output_path, output_fourcc,cap.get(5), (CROP_HEIGHT, CROP_WIDTH))      

    while True:

        status, frame = cap.read() 
        
        if not status:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        faces = face_detector(gray) 

        if(len(faces)): 
            face = faces[0]   
            x, y, height, width = face.left(), face.top(), face.height(), face.width() 
            height = int(height*FACE_HEIGHT) 
            if(target_width is None or target_height is None):
                target_width = width  
                target_height = height 
            else:
                target_width = max(target_width, width) 
                target_height = max(target_height, height) 

            cropped_frame = cv2.resize(frame[y+height: y+(height*2), x: x+width], (target_width, target_height)) 
            cropped_frame = cv2.resize(cropped_frame, (CROP_WIDTH, CROP_HEIGHT))  
            cropped_frame = tf.image.rgb_to_grayscale(cropped_frame)     
            lip_frames.append(cropped_frame)  

        else:
            cropped_frame = np.zeros((CROP_WIDTH,CROP_HEIGHT,3))    

            # pass   

        if display:  
            try:    
                cv2.imshow("Lips", cropped_frame) 
                cv2.imshow("Normal",frame)  
            except Exception as e:
                print(e) 
                pass 
            
            if(cv2.waitKey(1) == ord('q')):
                break

        if output_path:   
            output.write(cropped_frame)  

         

    cap.release() 
    cv2.destroyAllWindows()  

    #processing Video Frames 

    mean = tf.math.reduce_mean(lip_frames)    
    std = tf.math.reduce_std(tf.cast(lip_frames, tf.float32))  

    processed = tf.cast(lip_frames - mean, tf.float32) / std 

    # return lip_frames  # Returns BGR IMAGE 
    # print(type(processed)) 

    # print(lip_frames) 
 

    # while True:
    #     for frame in processed.numpy():   
    #         cv2.imshow("Processed",frame)

    #     if cv2.waitKey(1) == ord('q'):
    #         break    

    # cv2.destroyAllWindows()          

    return processed[:75] #return only 75 Frames 

  
def frames_to_video(name, frames, gray = False):  
    fps = 25 
    out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (len(frames), len(frames[0])), not gray) 
    for i in range(75):
        try:
            out.write(frames[i]) 
        except:
            break 
    out.release()  




vocab = "abcdefghijklmnopqrstuvwxyz'?!123456789 " 
vocab = [x for x in vocab] 

char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token = "") 
num_to_char = tf.keras.layers.StringLookup(vocabulary = char_to_num.get_vocabulary(), oov_token = "", invert = True)  


def load_alignments(path:str):
    with open(path, 'r') as f: 
        lines = f.readlines() 
        tokens = [] 
        #lst = [line.split(" ")[2].strip() for line in lines if line.strip()[-3:] != "sil"]
        for line in lines:
            line = line.split() 
            if(line[2] !='sil'):
                tokens = [*tokens," ",line[2]]  

    unicode_encoded = tf.strings.unicode_split(tokens, input_encoding = "UTF-8")  
    chars = tf.reshape(unicode_encoded, (-1))[1:] 

    return char_to_num(chars)  

# print(load_alignments("test.align"))  



def load_data(path: str): 
    path = bytes.decode(path.numpy())
    #file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_path = path.split("\\") 
    folder_name = path.split("\\")[-2]  
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join("GRID",folder_name,f'{file_name}.mpg')
    alignment_path = os.path.join("GRID",'alignments',folder_name,f'{file_name}.align') 
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments


if __name__ == "__main__":
    # test_video = load_video("test_video.mpg").numpy().tolist()   
    # print(len(test_video), len(test_video[0]), len(test_video[0][0]))  
    # print("\n\n") 
    # print(type(test_video))   
    # import json 
    
    # dic = {
    #     "input_data":[
    #         {
    #             "values":[test_video] 
    #         }
    #     ]
    # }

    # with open("sample.json",'w') as f:
    #     json.dump(dic, f)      

    # frames_to_video("face_mesh.mp4",landmark_facemesh("./Dataset/bbae4a.mp4"), gray = False)  

    # cap = cv2.VideoCapture("face_mesh.mp4"); 
    # while True:
    #     status, frame = cap.read()     
    #     if(not status):
    #         break 
    #     cv2.imshow("Mesh", frame) 
    # cap.release() 
    # cv2.destroyAllWindows()  
    pass 



