# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:46:50 2024

@author: chitv
"""


import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model = pickle.load(open('C:/Users\chitv/OneDrive/Desktop/Data_Science_Internship/3_Parkinsons Disease Detection/parkisons_test_data.sav','rb'))


#Creating a function for prediction

def parkinsons_prediction(input_data):
    

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person does not have Parkinsons Diseases'
    else:
        return 'The Person have a Parkinsons'
                          


def main():
    
    # given a title
    st.title('Parkinsons Diseases Prediction Web App')
    
    # getting the input from the user
    #name = st.text_input('Person Name: ')
    MDVP_Fo_Hz = st.text_input('MDVP_Fo_Hz: ')
    MDVP_Fhi_Hz = st.text_input('MDVP_Fhi_Hz: ')
    MDVP_Flo_Hz = st.text_input('MDVP_Flo_Hz: ')
    MDVP_Jitter_Per = st.text_input('MDVP_Jitter_Per: ')
    MDVP_Jitter_Abs = st.text_input('MDVP_Jitter_Abs: ')
    MDVP_RAP = st.text_input('MDVP_RAP: ')
    MDVP_PPQ = st.text_input('MDVP_PPQ: ')
    Jitter_DDP = st.text_input('Jitter_DDP: ')
    MDVP_Shimmer = st.text_input('MDVP_Shimmer: ')
    MDVP_Shimmer_dB = st.text_input('MDVP_Shimmer_dB: ')
    Shimmer_APQ3 = st.text_input('Shimmer_APQ3: ')
    Shimmer_APQ5 = st.text_input('Shimmer_APQ5: ')
    MDVP_APQ = st.text_input('MDVP_APQ: ')
    Shimmer_DDA = st.text_input('Shimmer_DDA: ')
    NHR = st.text_input('NHR: ')
    HNR = st.text_input('HNR: ')
    #status = st.text_input('status: ')
    RPDE = st.text_input('RPDE: ')
    DFA = st.text_input('DFA: ') 
    spread1 = st.text_input('spread1: ')
    spread2 = st.text_input('spread2: ')
    D2 = st.text_input('D2: ')
    PPE = st.text_input('PPE: ')
    
    
    
    #code for prediction 
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Parkinsons Diseases Test Result'):
        diagnosis = parkinsons_prediction([MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter_Per,MDVP_Jitter_Abs,MDVP_RAP,MDVP_PPQ,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_APQ,Shimmer_DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE])
        
    
    st.success(diagnosis)
    
    

if __name__=='__main__':
    main()


    
    
    
    
    
    
    
    
    
    
    



