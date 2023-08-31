import requests 

from utils import load_video, num_to_char  
import tensorflow as tf   

wml_credentials = {
    "url":"https://us-south.ml.cloud.ibm.com", 
    "apikey":"50tndXfHZWBvTOYzo-IG1MOK6LQAsSNObk0XgbdkBvSW" 
}

DEPLOYMENT_ID = "5015465f-770f-459a-9a06-c071fd198ffd"           # OLD_DEPLOYMENT_ID    #"fee21b8e-09d1-47fe-adb8-8d035b334cf2" 


# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.

API_KEY = "06OUEcO3iWSP-hDf_yAKIR3lE6MS6RTSmLFlmbtjNDd3"    # NEW_API_KEY       # "50tndXfHZWBvTOYzo-IG1MOK6LQAsSNObk0XgbdkBvSW"   # OLD_API_KEY


def watson_speech_prediction(frames):   

    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={
        "apikey":API_KEY, 
        "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'
    }) 

    status = False 

    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line

    payload_scoring = {
            "input_data": [
                {
                    "fields": [], 
                    "values": [ frames ]   
                }
            ]
        }  

    response_scoring = requests.post(f'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/{DEPLOYMENT_ID}/predictions?version=2021-05-01', json=payload_scoring,headers=header) 


    response = response_scoring.json()
    

    try:

        prediction = response['predictions'][0]["values"][0][0]    

        prediction = tf.expand_dims(prediction, axis = 0)  # Expanding shape to match input shape  

        decoded = tf.keras.backend.ctc_decode(prediction, input_length = [75], greedy = True)[0][0].numpy() 

        final_output = bytes.decode(tf.strings.reduce_join(num_to_char(decoded)).numpy()) 

        status = True 

    except Exception as e:
        errors = response["errors"] 
        if(errors) : 
            err_code = errors[0]["code"]  
            err_msg = errors[0]["message"] 

            print("Error Code:", err_code) 
            print("Error Message")   
            final_output = err_msg 
        else:
            final_output = "Unexpected Error Occured !" 

    return final_output, status