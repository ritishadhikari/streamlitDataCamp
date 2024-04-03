import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64

@st.cache_data()
def get_fvalue(val):
    feature_dict={"No":1, "Yes":2}
    for key,value in feature_dict.items():
        if val==key:
            return value



def get_value(val, my_dict:dict):
    for key, value in my_dict.items():
        if val==key:
            return value

app_mode=st.sidebar.selectbox(label="Select Page",
                              options=["Home","Prediction"])

if app_mode=="Home":
    st.title(body="Loan Predicition".upper())
    st.write("App reconstructed by Ritish Adhikari")
    st.image("loan_image.JPG") 
    st.markdown("Dataset :")
    data=pd.read_csv(filepath_or_buffer="loan_dataset.csv")
    st.write(data.head())
    st.markdown("Application Income Vs Loan Amount")
    st.bar_chart(data[['ApplicantIncome','LoanAmount']].head(20))
elif app_mode=="Prediction":
    st.image("prediction_header.JPG")
    st.subheader("Sir/Ma'am, You need to fill all necessary information in order to to know your loan request")
    st.sidebar.header("Informations about the Client:")
    gender_dict={"Male":1,"Female":2}
    feature_dict={"No":1,"Yes":2}
    edu={'Graduate':1,"Not Graduate":2}
    prop={"Rural":1,'Urban':2,"SemiUrban":3}

    ApplicationIncome=st.sidebar.slider(
        label='ApplicationIncome ($)',
        min_value=0,
        max_value=10000,
        value=3500
        )
    CoapplicantIncome=st.sidebar.slider(
        label='CoapplicantIncome ($)',
        min_value=0,
        max_value=10000,
        value=4500
        )
    LoanAmount=st.sidebar.slider(
        label="Loan Amount in K$",
        min_value=9,
        max_value=700,
        value=200
        )
    Loan_Amount_Term=st.sidebar.selectbox(
        label="Term Length of Loan in Months",
        options=(12,36,60,84,120,180,240,300,360)
    )
    Credit_History=st.sidebar.radio(
        label="Have you taken any loans earlier?",
        options=(0.0,1)
    )
    Gender=st.sidebar.radio(
        label="What's Your Gender?",
        options=tuple(gender_dict.keys())
    )
    Married=st.sidebar.radio(
        label="Are you Married ?",
        options=tuple(feature_dict.keys())
    )
    Self_Employed=st.sidebar.radio(
        label="Are you self employed?",
        options=tuple(feature_dict.keys())
        )
    Dependents=st.sidebar.radio(
        label="How many Dependents do you have?",
        options=["0","1","2","3+"]
        )
    Education=st.sidebar.radio(
        label="Education Qualification?",
        options=tuple(edu.keys())
        )
    Property_Area=st.sidebar.radio(
        label="Property Area for the Loan?",
        options=tuple(prop.keys())
    )
    
    class_0,class_1,class_2,class_3=(0,0,0,0)
    if Dependents=="0": class_0=1
    elif Dependents=="1": class_1=1
    elif Dependents=="2": class_2=1
    else: class_3=1

    Rural, Urban, SemiUrban = 0,0,0
    if Property_Area=="Urban": Urban=1
    elif Property_Area=="SemiUrban": SemiUrban=1
    else: Rural=1

    data1={
        "Gender": Gender,
        "Married": Married,
        "Dependents":[class_0,class_1,class_2,class_3],
        "Education": Education,
        "ApplicantIncome" : ApplicationIncome,
        "CoapplicantIncome":CoapplicantIncome,
        "Self Employed": Self_Employed,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term":Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Property_Area":[Rural,Urban,SemiUrban]
    }

    feature_list=[
        ApplicationIncome,CoapplicantIncome,LoanAmount,Loan_Amount_Term,
        Credit_History, get_value(val=Gender,my_dict=gender_dict),
        get_fvalue(val=Married),data1["Dependents"][0],data1["Dependents"][1],
        data1["Dependents"][2],data1["Dependents"][3], 
        get_value(val=Education,my_dict=edu),get_fvalue(val=Self_Employed),
        data1["Property_Area"][0],data1["Property_Area"][1],data1["Property_Area"][2]
        ]

    single_sample=np.array(feature_list).reshape(1,-1)

    if st.button(label="Predict"):
        with open(file="6m-rain.gif",mode="rb") as file_:
            contents=file_.read()
            data_url=base64.b64encode(s=contents).decode("utf-8")

        with open(file="green-cola-no.gif",mode="rb") as file:
            contents=file.read()
            data_url_no=base64.b64encode(s=contents).decode("utf-8")
        
        # Download this model
        loaded_model=pickle.load(open("Random_Forest.sav",mode="rb"))
        prediction=loaded_model.predict(single_sample)

        if prediction[0]==0:
            st.error('According to our Calculations, you will not get the loan from Bank')
            st.markdown(
            body=f'<img src="data:image/gif;base64,{data_url_no}" alt="cat gif">',
            unsafe_allow_html=True
            )
        
        elif prediction[0]==1:
            st.success('Congratulations!! you will get the loan from Bank')
            st.markdown(
            body=f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
            )
            
