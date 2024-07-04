import streamlit as st
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import joblib

# Bring the lung cancer prediction model
liver_model = joblib.load('liver_model.sav')
lung_cancer_model = joblib.load('lung_cancer_model.sav')
thyroid_model = joblib.load('LogisticRegression_thyroid.sav')

# Load the dataset
lung_cancer_data = pd.read_csv('survey lung cancer.csv')
thyroid_data = pd.read_csv('hypothyroid.csv')
thyroid_data["referral source"].replace({"SVHC": 0, "other": 1, "SVI": 2, "STMW": 3, "SVHD": 4}, inplace=True)

# convert "M" to 0 and "F" to 1 in the "gender" column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

# Adding the SurveyMonkey invitation script
st.write("If you'd like to give feedback about this application. :) [Click Here](https://www.surveymonkey.com/r/D7C2L9B)")                                                                   

with st.sidebar:
    st.title("Multi-Disease Classification")
    selected = st.selectbox("Select Disease Prediction", ["Liver Prediction", "Lung Cancer Prediction", "Thyroid Disease Prediction"])

# Lung cancer prediction page
if selected == 'Lung Cancer Prediction':
    st.title("Lung Cancer Prediction")
    image = Image.open('lung.jpg')
    st.image(image, caption='Lung Cancer Prediction')

    # User Inputs
    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
    with col2:
        age = st.number_input("Age")
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])

    with col2:
        anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])

    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])

    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])

    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    cancer_result = ''

# Prediction button
    if st.button("Predict Lung Cancer"):
        # create a dataframe with user inputs
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        # map string values to numeric
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

        # Strip leading and trailing whitespaces from column names
        user_data.columns = user_data.columns.str.strip()
      
        # convert columns to numeric where necessary
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Perform prediction
        cancer_prediction = lung_cancer_model.predict(user_data)

        # Display result
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
            image = Image.open('negative.jpg')
            st.image(image, caption='')

        st.success(name + ', ' + cancer_result)
        

# Liver prediction page
if selected == 'Liver Prediction':  
    st.title("Liver disease prediction")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver disease prediction.')

    name = st.text_input("Name:")
    col1, col2, col3 = st.columns(3)

    with col1:
        display = ("Male", "Female")
        options = list(range(len(display)))
        gender = st.selectbox("Gender", options, format_func=lambda x: display[x])
        gender = 1 if gender == 0 else 2
    with col2:
        age = st.number_input("Enter your age") # 2 
    with col3:
        Total_Bilirubin = st.number_input("Enter your Total_Bilirubin") # 3
    with col1:
        Direct_Bilirubin = st.number_input("Enter your Direct_Bilirubin")# 4

    with col2:
        Alkaline_Phosphotase = st.number_input("Enter your Alkaline_Phosphotase") # 5
    with col3:
        Alamine_Aminotransferase = st.number_input("Enter your Alamine_Aminotransferase") # 6
    with col1:
        Aspartate_Aminotransferase = st.number_input("Enter your Aspartate_Aminotransferase") # 7
    with col2:
        Total_Protiens = st.number_input("Enter your Total_Protiens")# 8
    with col3:
        Albumin = st.number_input("Enter your Albumin") # 9
    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Enter your Albumin_and_Globulin_Ratio") # 10 

    # Code for prediction
    liver_dig = ''

    if st.button("Liver disease test result"):
        # Prepare input data
        input_data = [[gender, age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, 
                       Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, 
                       Albumin, Albumin_and_Globulin_Ratio]]
        
        # Perform prediction
        liver_prediction = liver_model.predict(input_data)

        # Check the prediction result
        if liver_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            liver_dig = "I am sorry to say but it seems like you have liver disease."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            liver_dig = "Congratulations, you don't have liver disease."
        
        st.success(name + ', ' + liver_dig)


# thyroid disease prediction
if selected == 'Thyroid Disease Prediction':
    st.title("Thyroid Disease Prediction")
    image = Image.open('thyroid.jpg')
    st.image(image, caption='Thyroid Disease Prediction')

    name = st.text_input("Name:")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    sex = st.selectbox("Sex", ["Male", "Female"])

    on_thyroxine = st.selectbox("On Thyroxine", ['No', 'Yes'])
    query_on_thyroxine = st.selectbox("Query on Thyroxine", ['No', 'Yes'])
    on_antithyroid_medication = st.selectbox("On Antithyroid Medication", ['No', 'Yes'])
    sick = st.selectbox("Sick", ['No', 'Yes'])
    pregnant = st.selectbox("Pregnant", ['No', 'Yes'])
    thyroid_surgery = st.selectbox("Thyroid Surgery", ['No', 'Yes'])
    I131_treatment = st.selectbox("I131 Treatment", ['No', 'Yes'])
    query_hypothyroid = st.selectbox("Query Hypothyroid", ['No', 'Yes'])
    query_hyperthyroid = st.selectbox("Query Hyperthyroid", ['No', 'Yes'])
    lithium = st.selectbox("Lithium", ['No', 'Yes'])
    goitre = st.selectbox("Goitre", ['No', 'Yes'])
    tumor = st.selectbox("Tumor", ['No', 'Yes'])
    hypopituitary = st.selectbox("Hypopituitary", ['No', 'Yes'])
    psych = st.selectbox("Psych", ['No', 'Yes'])
    TSH_measured = st.selectbox("TSH Measured", ['No', 'Yes'])
    TSH = st.number_input("TSH")
    T3_measured = st.selectbox("T3 Measured", ['No', 'Yes'])
    T3 = st.number_input("T3")
    TT4_measured = st.selectbox("TT4 Measured", ['No', 'Yes'])
    TT4 = st.number_input("TT4")
    T4U_measured = st.selectbox("T4U Measured", ['No', 'Yes'])
    T4U = st.number_input("T4U")
    FTI_measured = st.selectbox("FTI Measured", ['No', 'Yes'])
    FTI = st.number_input("FTI")
    TBG_measured = st.selectbox("TBG Measured", ['No', 'Yes'])
    TBG = st.number_input("TBG")
    referral_source = st.selectbox("Referral Source", ["SVHC", "other", "SVI", "STMW", "SVHD"])
    referral_source_mapping = {"SVHC": 0, "other": 1, "SVI": 2, "STMW": 3, "SVHD": 4}
    
    if st.button("Predict Thyroid Disease"):

        yes_no_map = {'No': 0, 'Yes': 1}
        sex_map = {'Male': 0, 'Female': 1}
        user_data = pd.DataFrame({
            'age': [age],
            'sex': [sex_map[sex]],
            'on thyroxine': [yes_no_map[on_thyroxine]],
            'query on thyroxine': [yes_no_map[query_on_thyroxine]],
            'on antithyroid medication': [yes_no_map[on_antithyroid_medication]],
            'sick': [yes_no_map[sick]],
            'pregnant': [yes_no_map[pregnant]],
            'thyroid surgery': [yes_no_map[thyroid_surgery]],
            'I131 treatment': [yes_no_map[I131_treatment]],
            'query hypothyroid': [yes_no_map[query_hypothyroid]],
            'query hyperthyroid': [yes_no_map[query_hyperthyroid]],
            'lithium': [yes_no_map[lithium]],
            'goitre': [yes_no_map[goitre]],
            'tumor': [yes_no_map[tumor]],
            'hypopituitary': [yes_no_map[hypopituitary]],
            'psych': [yes_no_map[psych]],
            'TSH measured': [yes_no_map[TSH_measured]],
            'TSH': [TSH],
            'T3 measured': [yes_no_map[T3_measured]],
            'T3': [T3],
            'TT4 measured': [yes_no_map[TT4_measured]],
            'TT4': [TT4],
            'T4U measured': [yes_no_map[T4U_measured]],
            'T4U': [T4U],
            'FTI measured': [yes_no_map[FTI_measured]],
            'FTI': [FTI],
            'TBG measured': [yes_no_map[TBG_measured]],
            'TBG': [TBG],
            'referral source': [referral_source_mapping[referral_source]]
        })
# perform prediction
        thyroid_prediction = thyroid_model.predict(user_data)
# display prediction result
        if thyroid_prediction[0] == 1:
            st.success(f"{name}, the model predicts that you have thyroid disease.")
        else:
            st.success(f"{name}, the model predicts that you do NOT have thyroid disease.")
