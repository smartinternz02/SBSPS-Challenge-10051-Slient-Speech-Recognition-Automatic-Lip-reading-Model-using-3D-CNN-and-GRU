# SBSPS-Challenge-10051-Slient-Speech-Recognition-Automatic-Lip-reading-Model-using-3D-CNN-and-GRU
Slient Speech Recognition : Automatic Lip reading Model using 3D CNN and GRU

Install the necessary requirements with  "pip install -r requirements.txt"

Used python version -> 3.9.17 

conda syntax : "conda create --name <environment_name> python==3.9 anaconda"    

Go to the WebApp folder and run "streamlit run app.py" to execute the website with streamlit 

Under the models folder there contains different model checkpoints in which  "best_model" is the latest training checkpoint this produced maximum effeciency as per our validation. 

NOTE : only 20 Computational Unit Hours are allowed in Lite plan of IBM Watson Machine Learning, Thus kindly use the resource wisely as API rejection occurs once 20CUH per month limit crosses (it takes around 0.5 CUH per request!)

In case of crossing 20CUH limit create your own deployment by following the steps 👇 

To create your own deployment : 
      1) Create a IBM Watson Studio and IBM Watson machine learning resource (cloud.ibm.com) 
      2) Under watson_deployment.ipynb alter the credentials accordingly (API KEY, SPACE ID etc.,) and execute the cells 
This creates a new deployment space in your ibm cloud 
Also modify the backend.py file (under webApp folder) accordingly to your new credentials (API KEY, DEPLOYMENT ID etc.,)  

NOTE : In case of dlib build fail, ensure that Microsoft Visual Studio 2019 C++ distributables are properly install 
      
      Try installing through conda by : "conda install dlib" 

