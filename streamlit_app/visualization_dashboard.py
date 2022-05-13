
import streamlit as st
import pandas as pd
import os

os.getcwd()
st.set_page_config(layout='wide')
cwd = os.path.abspath('')
files = os.listdir(cwd)

st.header("Select Results file")
data = st.file_uploader("", type='xlsx')
if data is not None:
    all_sheet = pd.ExcelFile(data)
    sheet_name = all_sheet.sheet_names
    col1, col2, col3, col4 = st.columns(4)

    # Select box to choose the feature
    with col1:
        Select_Features = st.selectbox('Choose any feature ',sheet_name)
    sheet_name= pd.read_excel(data, Select_Features)
    # st.write(sheet_name)

    #Create the select box to choose scaler
    uniqueValues = sheet_name['normal_type'].unique()

    with col3:
        Select_Scaler = st.selectbox('Choose Normalize Technique ',uniqueValues)
    # st.write(Select_Scaler)

    #create the select box to choose the Name of store
    uniqueValues_1= sheet_name['store_name'].unique()
    with col2:
        Select_store = st.selectbox("Choose the store Name",uniqueValues_1)
    # st.write(uniqueValues_1)

    uniqueValues_2= sheet_name['algorithm'].unique()
    with col4:
        Select_algo= st.selectbox("Choose Algorithm",uniqueValues_2)

    # selecting the store name and scaler name according to choice:
    store= sheet_name[sheet_name["store_name"]==Select_store]
    scaler= store[store["normal_type"]==Select_Scaler]
    algo= scaler[scaler["algorithm"]==Select_algo]

    data = pd.DataFrame(algo)

    if 'camera_count' in data:
        df = data[['time','camera_count', 'predicted_cal']]
    else:
        df = data[['time','total', 'predicted_cal']]
    
    df = df.rename(columns={'time':'index'}).set_index('index')
    st.line_chart(df)
    st.write("\n")


    col1, col2, col3 = st.columns([1,6,1])

    with col1:
        st.write("")

    with col2:
        st.dataframe(data=data.drop(['store_name', 'algorithm', 'normal_type'], axis=1), width=20000, height=500)

    with col3:
        st.write("")





