import pickle
import streamlit as st
from streamlit_option_menu import option_menu

#loading models
diabetes_model=pickle.load(open("C:/Users/hp/Desktop/code/multiple disease/Multiple-Disease-Prediction/diabetes.sav",'rb'))
parkison_model=pickle.load(open("C:/Users/hp/Desktop/code/multiple disease/Multiple-Disease-Prediction/parkison.sav",'rb'))
heart_model=pickle.load(open("C:/Users/hp/Desktop/code/multiple disease/Multiple-Disease-Prediction/heart.sav",'rb'))
scaler_diabetes=pickle.load(open("C:/Users/hp/Desktop/code/multiple disease/Multiple-Disease-Prediction/scaler_dia.pkl",'rb'))
scaler_parkison=pickle.load(open("C:/Users/hp/Desktop/code/multiple disease/Multiple-Disease-Prediction/scaler_par.pkl",'rb'))

#sidebar for navigate
with st.sidebar:
    selected=option_menu('Multiple disease prediction System',
                         ['Diabetes Prediction',
                          'Heart-disease Prediction',
                          'Parkison disease prediction'],
                          icons=['activity','heart','file-medical'],
                          default_index=0)
    
    #diabetes page
if(selected=='Diabetes Prediction'):
    st.title('Diabetes prediction using ML')

    col1, col2, col3 = st.columns(3)    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')        
    with col2:
        Glucose = st.text_input('Glucose Level')    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')    
    with col2:
        Insulin = st.text_input('Insulin Level')    
    with col3:
        BMI = st.text_input('BMI value')    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')    
    with col2:
        Age = st.text_input('Age of the Person')
        #code for prediction
        diab_output=''
        if st.button('Diabetes Test Result'):
            scaled_data=scaler_diabetes.transform([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            diab_pred=diabetes_model.predict(scaled_data)

            if(diab_pred[0]==1):
                diab_output='Pearson is diabetic'
            else:
                diab_output='Person is not diabetic'

        st.success(diab_output)


#heart page
if(selected=='Heart-disease Prediction'):
    st.title('Heart-disease Prediction using ML')

    col1, col2, col3 = st.columns(3)    
    with col1:
        age = st.text_input('Age')        
    with col2:
        sex = st.text_input('Sex')        
    with col3:
        cp = st.text_input('Chest Pain types')        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')        
    with col3:
        exang = st.text_input('Exercise Induced Angina')        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')      
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)

#parkisons page
if(selected=='Parkison disease prediction'):
    st.title('Parkison disease prediction using ML')

    col1, col2, col3,= st.columns(3) 
    
    with col1:
        fo = st.text_input('MDVP_Fo(Hz)')        
    with col2:
        fhi = st.text_input('MDVP_Fhi(Hz)')        
    with col3:
        flo = st.text_input('MDVP_Flo(Hz)')        
    with col1:
        Jitter_percent = st.text_input('MDVP_Jitter(%)')       
    with col2:
        Jitter_Abs = st.text_input('MDVP_Jitter(Abs)')        
    with col3:
        Shimmer = st.text_input('MDVP_Shimmer')        
    with col1:
        NHR = st.text_input('NHR')        
    with col2:
        HNR = st.text_input('HNR')        
    with col3:
        RPDE = st.text_input('RPDE')        
    with col1:
        DFA = st.text_input('DFA')        
    with col2:
        spread1 = st.text_input('spread1')        
    with col3:
        spread2 = st.text_input('spread2')        
    with col1:
        D2 = st.text_input('D2')      
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        scaled_data=scaler_parkison.transform([[fo, fhi, flo, Jitter_percent, Jitter_Abs,Shimmer,NHR,HNR,RPDE,DFA,spread1,spread2,D2]])
        parkinsons_prediction = parkison_model.predict(scaled_data)                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)