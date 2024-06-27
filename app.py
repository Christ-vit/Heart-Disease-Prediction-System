import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
linRegModel = LogisticRegression()

# Label Encoding for categorical features
label_encoder = LabelEncoder()

st.title('My First Streamlit App')

def load_machineLearning_model():
    with open('trainedModel.pkl', 'rb') as modelFile:
        data = pickle.load(modelFile)
    return data

data = load_machineLearning_model()

model_Loaded = data["model"]
encoding_Loaded = data["labelEnc_Mechansims"]

def Display():
    st.title('Heart Disease Prediction System')
        
    st.write("###Fill in some information so that we can provide you with a prediction ###")
        
    sexOption = (
    "male", 
    "female"
    )

    cpOption = (
    "typical angina", 
    "atypical angina",
    "non-anginal pain",
    "asymptomatic",
    )

    
    
    fbsOption = (
    "false", 
    "true"
    )

    restecgOption = (
    "normal", 
    "abnormal",
    "venticular hyptertrophy"
    )

    
    exangOption = (
    "no", 
    "yes"
    )
    
    slopeOption = (
    "upsloping", 
    "flat", 
    "downsloping"
    )

    caOption = (
    "mild", 
    "moderate", 
    "severe"
    )

    thalOption = (
    "normal", 
    "fixed defect",
    "reversible defect"
    )
    
    ageMin = 0
    ageMax = 150
    notNegative = 0
    
    ageValue = st.number_input('Enter your age: ', min_value=ageMin, max_value=ageMax, value=0, step=1, format='%d')
    
    sexValue = st.selectbox("Select your gender", sexOption)
    
    cpValue = st.selectbox("Select your chest pain type", cpOption)
    
    trestpsValue = st.number_input('Enter your resting blood pressure (mm Hg): ', min_value=notNegative, value=0, format='%d')
    
    cholValue = st.number_input('Enter your serum cholestral pressure in mg/dl: ', min_value=notNegative, value=0,format='%d')
    
    fbsValue = st.selectbox("Is your fasting sugar  greater than 120 mg/dl ?", fbsOption)
    
    restecgValue = st.selectbox("Select your resting electrocardiographic results", restecgOption)
    
    thalachValue = st.number_input('Enter your maximum heart rate achieved: ', min_value=notNegative, value=0,format='%d')

    exangValue = st.selectbox("Exercsie induced angina ?", exangOption)
    
    oldpeakValue = st.number_input('Enter your ST depression induced by exercise relative to rest: ', min_value=notNegative, max_value=10.0, value=0, step=0.1)
    
    slopeValue = st.selectbox("Select the slope peak exercise ST segment", slopeOption)
    
    caValue = st.selectbox("Number of major vessels coloured by fluoroscopy ?", caOption)
    
    thalValue = st.selectbox("Status of the heart", thalOption)
    
    submitButton = st.button("Provide prediction")
    
    if submitButton:
        if ageValue < ageMin or ageValue > ageMax:
            st.error(f'Error: Age must be between {ageMin} and {ageMax}.')
        else:
            if notNegative > trestpsValue and  notNegative > cholValue and notNegative > thalachValue and notNegative > oldpeakValue:
                st.error(f'Those fields cannot be negative')
            else:
                #Capture values into an array for encoding according to the respective positions of the columns
                x = np.array([[ageValue, sexValue, cpValue,trestpsValue, cholValue,fbsValue,restecgValue, thalachValue,exangValue,oldpeakValue,slopeValue,caValue,thalValue]])
                
                #Encode the values using the loaded label encoders
                x[:, 1] = label_encoder.fit_transform(x[:, 0])
                x[:, 2] = label_encoder.fit_transform(x[:, 2])
                x[:, 5] = label_encoder.fit_transform(x[:, 5])

                x[:, 6] = label_encoder.fit_transform(x[:, 6])
                x[:, 8] = label_encoder.fit_transform(x[:, 8])
                x[:, 10] = label_encoder.fit_transform(x[:, 10])
                x[:, 11] = label_encoder.fit_transform(x[:, 11])
                x[:, 12] = label_encoder.fit_transform(x[:, 12])

                #convert the encoded values to floats 
                x=x.astype(float)
                
                
                # Make the prediction using the loaded model
                resultPrediction = linRegModel.predict(x)
                
                # Interpret the prediction result
                if resultPrediction[0] == 0:
                    st.subheader("You are healthy.")
                else:
                    st.subheader("You are not healthy.")
Display()