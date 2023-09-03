import streamlit as st 
#import time 
#import asyncio 
#import cv2 
from moviepy import editor as moviepy 
import imageio 

#import pandas as pd 
#import numpy as np 

import os 

#from matplotlib import pyplot as plt 
#import plotly.express as px 

from utils import load_video, num_to_char 

 
from streamlit_option_menu import option_menu 

from backend import watson_speech_prediction, speech_prediction  



st.set_page_config(layout = "wide", initial_sidebar_state="expanded", page_title = "LipNet - alphaAxon", page_icon="./assets/favicon.png")  

st.session_state["video_path"] = None 

#  Bootstrap CDN
#  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" integrity="sha384-rbsA2VBKQhggwzxH7pPCaAqO46MgnOM80zW1RWuH61DGLwZJEdK2Kadq2F9CUG65" crossorigin="anonymous">
#

with st.sidebar:
    st.image("assets/alphaAxon_final.png", use_column_width = True)  

CDNs = """         
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@100;300;400;600&display=swap" rel="stylesheet">
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Poppins:wght@100;300;400;600&display=swap" rel="stylesheet">

            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;500;700&family=Orbitron:wght@400;500;600;700&family=Poppins:wght@100;300;400;600&display=swap" rel="stylesheet">
        """

st.markdown(CDNs, unsafe_allow_html=True) 



navbar = option_menu(menu_title = None, 
            options = ["Home", "LipNet Model","About"], 
            icons = ["home", "brain", "search"], 
            default_index = 0, orientation="horizontal")    


if navbar == "Home":

    st.title("A Stunning Approach to Capture Speech") 

    st.markdown("""

        <div class="container">
            <div class="left">
                <h4 class = "space" ><b> Transforming Gestures into Crystal Clear Communication </b></h4>
                <h3 class = "space">Uncover the hidden melodies through our captivating Lip reading platform</h3> 
            </div>
            <div class="middle">
                <h3 class = "space">Presenting you the power of <br> <b class = "bold">Computer Vision </b> and <b class = "bold"> Neural Networks </b></h3> 
                <i><h4>"See the Unheard, Understand the Unsaid."</h4></i> 
            </div>
            <div class = "right">
                <h3 class = "space">Unlocking cutting-edge possibilities with Deep Learning</h3>
                <h3 class = "thin">Revolutionizing Silent Speech Analysis for a Connected World</h3>   
            </div>
        </div>

        <div class = "container container-two"> 
                <div class = 'neo'> 
                    <h3 class = "open-sans">Immense yourself in the muted elegance of Lip-reading</h2> 
                    <h4>Open up new possibilities of communication with intelligence.</h4>
                    <h4 class = "open-sans" style = "line-height:1.5;" >"Uncover hidden melodies of non-verbal communication through our captivating silent lip reading platform"</h4> 
                </div>
                <div>
                    <button class = "hero-button">Explore The Speech Model Now! </button>
                </div>
        </div> 
    
    """, unsafe_allow_html = True)  

    with st.sidebar:
        st.write("---") 
        st.markdown("""
            <p>This Technology is Extremely useful in environments where audio isn't optimal for recognition</p> 
            <p>Can be used in Medical Diagnosis for people with disability, who lost their abikity to speak</p> 
            <p>Can be used to improve Human-Computer interface, wearables like augmented reality could open up new possibilities of interaction and control</p> 
        """, unsafe_allow_html=True)  




elif navbar == "LipNet Model": 

    st.sidebar.write("---") 
    st.sidebar.info("A little insight About LipNet")       
    st.sidebar.markdown("""
        <p style = "font-size:1.1rem;">Our model is trained on delicate data with optimal lighting throught the face and face pointing direcly at the camera, Thus as of now it might not be so accurate with every type of video</p> 
        <p style = "font-size:1.1rem;">Also the video should be exactly 75 frames, that's the input shape our model can process</p>
        <p style = "font-size:1.1rem;">Hence we've provided a part of prepeocessed data, which can be used to test the model.</p>
    """, unsafe_allow_html = True)   
    st.sidebar.warning("Hence it's reccomended to use provided dataset for evaluation")  
    st.title("The Ultimate Deep Learning Model to read :lips:")  
    st.title(" ")  

    st.info("Note : Our Model is not so mature, yet we've been constantly working on this to improve its abilities, Thus be kind with it :heart: | \n Watson Machine Learning supports only 20CUH of free computation use the resource wisely", icon="📝") 

    st.title(" ") 

    gap1,col1, gap2,col2, gap3 = st.columns([1,10,1,10,1])  
    dataset = os.listdir("./Dataset")
    
    

    video_path = col1.selectbox(label = "Select Video from existing Dataset: ", options = ["Select a Video",*dataset]) 

    if(video_path != "Select a Video" and (video_path.endswith(".mp4") or video_path.endswith(".mpg"))): 
        clip = moviepy.VideoFileClip("./Dataset/"+video_path)    
        clip.write_videofile("./temp/test.mp4")  
        col1.video("./temp/test.mp4") 
        st.session_state["video_path"] = "./temp/test.mp4" 

    if(st.session_state.video_path is None):
        col2.markdown("""
            <div class = "box preprocess">
                <h2>Select a Video to Process into tensors</h2> 
                <h3>By feature extraction only lower face will be detected and converted into standardized form</h3>
            </div>
        """,unsafe_allow_html = True) 

    elif st.session_state.video_path:


        frames = load_video(st.session_state.video_path).numpy()  
        #st.write(type(frames)) 
        #st.write(frames)     
        col2.markdown("""
            <h3 style = "font-size:1.7rem; margin-top:2.5rem;">This is What the Deep Learning Model Sees! </h3> 
            <h4 style = "font-weight:500;  margin-bottom:2.5rem; ">(No sound, Only some bare black-and-white pixels ! )</h4> 
        """, unsafe_allow_html = True)   
        #frame_index = col2.slider("",0,75,20)     
        #df_frames =  pd.DataFrame([range(0,75), np.array([frames.reshape(1,75,46,140)]) ], columns = ["idx", "frame"])  
        #img = px.imshow(df_frames, animation_frame = df_frames["idx"])  
        #img = px.imshow(frames[frame_index].reshape(46,140))   
        #col2.plotly_chart(img)     

        imageio.mimsave("./temp/anim.gif",frames, format = "gif" ,fps = 20)    
        col2.image("./temp/anim.gif", use_column_width=True)   
  

        # subcol1, subcol2 =  col2.columns(2)
        
        # for frame_index in range(0,35,15): 
        #     img = px.imshow(frames[frame_index].reshape(46,140))     
        #     subcol1.plotly_chart(img, use_container_width = True)      
        #     img = px.imshow(frames[frame_index + 15].reshape(46,140))     
        #     subcol2.plotly_chart(img, use_container_width = True)  
        
        #  def predict_speech(): 

        st.markdown("""
            <h2 class = "center-text" >Prediction From the Model</h2>
            """,unsafe_allow_html=True)

        with st.spinner("Predicting..."): 
            # comment the watson_speech_prediction() and uncomment speech_prediction() in case of CUH error 
            prediction, status = speech_prediction(frames)  #watson_speech_prediction(frames.tolist()) 

        if status:
            st.success("Speech SuccessFully Predicted :sparkles: ") 
            st.markdown(f"""
                <h3 class = "center-text">Predicted Words : <b><i>{prediction}</i><b> </h3>  
            """,unsafe_allow_html = True) 
        else:
            st.error(prediction+" Please retry after sometime") 
            st.write("Refer app.py line 175 for fixing this error...")   
        st.warning("The Model isn't always 100\% Accurate, that's what makes it more humanly :wink: :smiling_face_with_smiling_eyes_and_hand_covering_mouth: ")   

        #predict_btn = col2.button("Predict Speech", on_click = predict_speech, type = "primary")    
    


elif navbar == "About":
    st.markdown(f"""
        <div class="about-container">
            <h1>About Us! : Team AlphaAxon</h1><br> 
            <h5>We the students of</h5>
            <h3>Dr. M.G.R Educational and Research Institute</h3> 
            <h5>Maduravoyal, Chennai, TamilNadu, India</h5> 
            <h3 style = "margin-top:1.5rem; ">Have done this Project under IBM Hack Challenge 2023</h3>
            <h4>Silent Speech Recognition with CNN, LSTM and Computer Vision.</h4><br> 
            <div class = "about_info"><span class = "">This Model is built with <b>Tensorflow</b> and trained by the <b>GRID dataset</b>,<br>which consists of videos and annotations of people uttering few words, and is Hosted by <br> <b>IBM Watson Machine Learning</b> and <b>IBM Cloud</b></span></div>  
            <div class = "footer">
                <div>
                <p>Built by :</p>   
                <p>Gurubaravikkram K R</p>
                <p>Dinesh Kumar S</p>
                <p>Hemanth Kumar M</p>
                <p>Harishwar J</p> 
                </div> 
            </div> 
        </div>
        
""", unsafe_allow_html = True) 
    
    with st.sidebar:
        st.markdown("""
            <p style = "font-size:1.2rem; margin-bottom:2rem; " >This Model is Based Upon the LipNet paper which contains of implementation of silent speech recognition using only visual data. </p><br> 
            <p style = "font-size:1.1rem">Various Deep Learning Techniques Such as Convolutional 3D Neural Networks and LSTM are used to construct the model. </p>
        """, unsafe_allow_html=True)   
        

# with open("style.css", 'r') as f:
#     styles = f"""<style>{f.readlines()}</style>"""  
#     st.markdown(styles, unsafe_allow_html=True)   

styles = """
    <style> 

        *{
            font-family:"Source Sans Pro", sans-serif; 
        }

        .open-sans{
            font-family: 'Open Sans', sans-serif; 
        }

        h1{
            display:block; 
            text-align:center; 
            font-weight:500; 
            letter-spacing:0.4rem;  
        }

        .space{
            line-height:2; 
        }

        .thin{
            font-weight:500; 
        }


        *{
            font-family:"Monospace", sans-serif; 
        }

        #app{
            font-size:3rem;  
        }

        .logo{
            width:100%; 
            height:auto; 
            display:flex; 
            justify-content:center; 
            align-items:center; 
        }

        .logo img{
            height: auto; 
            width: 350px; 
        }

        header, footer{
            visibility:hidden;        
        }

        .justify-center{
            display:flex; 
            flex-direction:row; 
            justify-content:center; 
        }

        div .block-container{ 
            padding:2rem; 
            padding-top:1rem; 
        }

        .container{
            width: 100%; 
            display: flex; 
            flex-direction:row; 
            justify-content:center; 
            margin-top:3rem; 
            gap:2rem; 
            text-align:center; 
            flex-wrap:wrap; 
        }

        .center-text{
            text-align:center; 
        }

        .hero-button{
            padding:5px 15px; 
            outline:none; 
            border:none; 
            border-radius:5px; 
            font-size:1.2rem; 
            background: #12c2e9;  /* fallback for old browsers */
            background: -webkit-linear-gradient(to right, #c471ed, #12c2e9);  /* Chrome 10-25, Safari 5.1-6 */
            background: linear-gradient(to right, #c471ed, #12c2e9); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */

        }

        .left, .middle, .right{
            width:30%;
            line-height:5px ; 
            letter-spacing:2px; 
            border-radius:2rem; 
            padding: 1rem 2.5rem; 
            box-shadow:5px 5px 10px 0px rgba(0,0,0,0.5); 
            min-width:300px; 
        }

        .left{
            background: #ee9ca7;  /* fallback for old browsers */
            background: -webkit-linear-gradient(to right, #ffdde1, #ee9ca7);  /* Chrome 10-25, Safari 5.1-6 */
            background: linear-gradient(to right, #ffdde1, #ee9ca7);
        }

        .middle{
            
            background: #757F9A;  /* fallback for old browsers */
            background: -webkit-linear-gradient(to right, #D7DDE8, #757F9A);  /* Chrome 10-25, Safari 5.1-6 */
            background: linear-gradient(to right, #D7DDE8, #757F9A); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */

        }

        .right{
            background: #36D1DC;  /* fallback for old browsers */
            background: -webkit-linear-gradient(to right, #5B86E5, #36D1DC);  /* Chrome 10-25, Safari 5.1-6 */
            background: linear-gradient(to right, #5B86E5, #36D1DC); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */

        }

        h3{
            font-weight:500; 
        }

        h4{
            font-weight:400;   
        }

        .bold{
            font-weight:700; 
        }

        .container-two{
            padding-top:2rem; 
            align-items:center; 
            gap:4rem;  
        }

        .container-two .neo{
            padding:1rem; 2rem;  
            border-radius:5px; 
            /*box-shadow:2px 2px 5px 2px rgba(0,0,0,0.3),  
                        -5px -5px 5px 0px rgba(255,255,255,.5), 
                        -2px -2px 5px 5px rgba(255,255,255,0.5);   */ 
            border-radius:1rem;  
            text-align:right; 
            width:60%;   
        }

        .about-container{
            display:flex; 
            flex-direction:column; 
            text-align:center; 
            margin-top:3rem; 
            align-items:center; 
        }

        .about-container h1{ 
            letter-spacing:2px; 
            font-weight:700; 
            font-family:"Poppins",sans-serif; 
        }

        .info{
            background:rgba(0,200,200,.4); 
            padding:1rem 2rem; 
            border: 2px solid lightblue;  
            border-radius:10px; 
            width:max-content;  
        }

        .about_info{
            font-weight:500; 
            line-height:2; 
            padding:2rem 2.5rem; 
            border-radius:10px;  
            font-size:1.3rem;  
            width:max-content; 
            margin-bottom:3rem; 
            background: #FFAFBD;  /* fallback for old browsers */
            background: -webkit-linear-gradient(to right, #ffc3a0, #FFAFBD);  /* Chrome 10-25, Safari 5.1-6 */
            background: linear-gradient(to right, #ffc3a0, #FFAFBD); /* W3C, IE 10+/ Edge, Firefox 16+, Chrome 26+, Opera 12+, Safari 7+ */

            color:black; 
            box-shadow:5px 5px 10px 0px rgba(0,0,0,0.5);  
        }

        .about-container .footer{
            width:100%; 
            display:flex; 
            justify-content:center;    
            align-items:center; 
            gap:2rem; 
            margin:2rem;  
            margin-right:3rem;    
           
        }

        .footer div{
            display:flex; 
            align-items:center; 
            justify-content:center; 
            padding:0.75rem 1.5rem;  
            gap:2rem; 
            background:#353b48;  
            color:white; 
            font-weight:700;  
            border-radius:1rem; 
        }

        .about-container .footer p{
            font-size:1.5rem; 
            text-transform:uppercase; 
            margin:0;   
        }

        .preprocess{
            display:flex; 
            flex-direction:column; 
            justify-content:center; 
        }

    </style> 

"""

logo_container = """
    <div class = "logo"> 
        <img  src="resized.png" ></img>   
    </div>
"""

st.markdown(styles, unsafe_allow_html=True)  


#my_grid = grid(1,[1,1], vertical_align = "top")     


    
    #first, center, last = st.columns([1,2,1])   
    #st.image(logo, use_column_width=True) 
    # st.markdown(logo_container, unsafe_allow_html=True)  
    # upload_file = st.file_uploader("Upload Video file to detect Words", type = ["mpg"])   
      
      




# def get_video(FILE_NAME): 
#     cap = cv2.VideoCapture(FILE_NAME) 
#     frames = [] 
#     image_container = st.image([])  

#     while True:  
#         status, frame = cap.read()

#         if not status:
#             break

#         # if not status:
#         #     cap = cv2.VideoCapture(FILE_NAME) 
#         #     status, frame = cap.read() 
          
#         img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
#         frames.append(img)   
#         # image_container.image(img)
    
#     return frames

# def process_video(frames):
#     processed = []  
#     for frame in frames:
#        processed.append(tf.image.rgb_to_grayscale(frame))    
#     return ( tf.cast(processed - tf.math.reduce_mean(processed), tf.float32) / tf.math.reduce_std(tf.cast(processed, tf.float32)))    
      


# run = st.checkbox("Run Video") 

# if run:
#     frames = get_video("test_4.mp4")

#     print("#"*10, "\n"*5)  
#     print(np.array(frames).shape)   
#     print(np.array(frames).dtype)    
#     print("#"*10, "\n"*5)

#     frames = tf.cast(process_video(frames)[:75], tf.int8)   

#     print("#"*10, "\n"*5)  
#     print(frames.shape)  
#     print("#"*10, "\n"*5)  
#     print(frames)   
    

#     #imageio.mimsave("animation.gif",frames, duration=20)  
#     image_container = st.image([])  
#     while run:
#         for i in frames.numpy(): 
#             image_container.image(i)  
#     # st.image("animation.gif", width = 400)      




# st.snow() 

# st.balloons() 







