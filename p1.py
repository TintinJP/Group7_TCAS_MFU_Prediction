import streamlit as st
import pandas as pd
import pickle

st.markdown(
    """
<style>
.body{
   background-color: gray;
}
.reportview-container .markdown-text-container {
    font-family: monospace;
}
.sidebar .sidebar-content {
    background-image: linear-gradient(aqua,gold,salmon);
    color: black;
    font-size: 40px;
}
.Widget>label {
    color:black;
    font-family: monospace;
}
[class^="st-b"]  {
    color: black;
    font-family: monospace;
}
.st-bb {
    background-color: transparent;
}
.st-at {
    
}
footer {
    font-family: monospace;
}
.reportview-container .main footer, .reportview-container .main footer a {
    color:#fcfafa;
}
header .decoration {
    background-image: none;
}


</style>
""",
    unsafe_allow_html=True,
)
st.image('./007.png')
st.header('TCAS MFU God7')

st.sidebar.header('User Input')
st.sidebar.subheader('Please enter your data:')


def get_input():
    #widgets
    k_Sex = st.sidebar.radio('Sex', ['Male','Female'])
    k_Stu = st.sidebar.radio('StudentType', ['FOREIGN','LOCAL'])
    k_Fac = st.sidebar.selectbox('FacultyName', ['School of Agro-industry',
       'School of Cosmetic Science', 'School of Dentistry',
       'School of Health Science', 'School of Information Technology',
       'School of Integrative Medicine', 'School of Law',
       'School of Liberal Arts', 'School of Management', 'School of Medicine',
       'School of Nursing', 'School of Science', 'School of Sinology',
       'School of Social Innovation'])
    k_Nat = st.sidebar.selectbox('NationName', ['AMERICA',
       'AUSTRALIA', 'BANGLADESH', 'BHUTAN', 'BRAZIL', 'CAMEROON', 'CHINA',
       'ENGLAND', 'FRENCH', 'INDIA', 'INDONESIA', 'JAPAN', 'LAO', 'MALAYSIA',
       'MYANMAR', 'PHILIPPINE', 'SINGAPORE', 'SOUTH AFRICA', 'SOUTH KOREA',
       'TAIWAN', 'THAI'])
    k_GPX = st.sidebar.slider('GPAX', 0.010, 4.000, 3.280)
    k_Eng = st.sidebar.slider('GPA_Eng', 0.750, 4.000, 3.370)
    k_Mat = st.sidebar.slider('GPA_Math', 0.450, 4.000, 2.817093)
    k_Sci = st.sidebar.slider('GPA_Sci', 0.620, 4.000, 3.030)
    k_Sco = st.sidebar.slider('GPA_Sco', 0.700, 4.000, 3.550)

   #  if k_Sex == 'Male': k_Sex = 'M'
   #  else: k_Sex = 'F'
    

    #dictionary
    data = {'Sex': k_Sex,
            'FacultyName':k_Fac,
            'StudentType': k_Stu,
            'NationName': k_Nat,
            'GPAX': k_GPX,
            'GPA_Eng': k_Eng,
            'GPA_Math': k_Mat,
            'GPA_Sci': k_Sci,
            'GPA_Sco': k_Sco
            }

    #create data frame
    data_df = pd.DataFrame(data, index=[0])
    return data_df

df = get_input()
st.write(df)

data_sample = pd.read_csv('TH.csv')
data_sample = data_sample.drop(columns=['Status'])
df = pd.concat([df, data_sample],axis=0)
st.write(df)


cat_data = pd.get_dummies(df[['Sex','StudentType','FacultyName','NationName']])
# cat_data = pd.get_dummies(df[[]])
# cat_data = pd.get_dummies(df[[]])
# cat_data = pd.get_dummies(df[[]])

st.write(cat_data)

#Combine all transformed features together
X_new = pd.concat([cat_data, df], axis=1)
X_new = X_new[:1] # Select only the first row (the user input data)

#Drop un-used feature
X_new = X_new.drop(columns=['Sex','StudentType','FacultyName','NationName'])
st.write(X_new)

# -- Reads the saved normalization model
load_nor = pickle.load(open('normalization.pkl', 'rb'))
#Apply the normalization model to new data
X_new = load_nor.transform(X_new)
st.write(X_new)

load_knn = pickle.load(open('best_knn.pkl', 'rb'))
# Apply model for prediction
prediction = load_knn.predict(X_new)
st.write(prediction)
