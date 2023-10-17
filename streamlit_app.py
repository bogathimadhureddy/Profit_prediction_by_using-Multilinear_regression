import pandas as pd
import streamlit as st 
# import numpy as np
from statsmodels.tools.tools import add_constant
from sqlalchemy import create_engine
import pickle, joblib

model1 = pickle.load(open('startup.pkl','rb'))
preprocess = joblib.load('preprocess.pkl')


def predict_MPG(startup, user, pw, db):

    engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

    startup.columns = 'RD','admin','marketing','state','profit'
    startup = startup.drop(['profit'],axis =1) # for new data this step is no need.

    cleandata = pd.DataFrame(preprocess.transform(startup),columns=preprocess.get_feature_names_out())
    cleandata = add_constant(cleandata)
    cleandata.drop(['categ__state_Florida','num__admin','categ__state_New York'],axis =1,inplace = True)

    prediction = pd.DataFrame(model1.predict(cleandata), columns = ['Profit'])
    
    final = pd.concat([prediction,startup], axis = 1)
    final.to_sql('mpg_predictons', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

    return final



def main():
    

    st.title("Fuel Efficiency prediction")
    st.sidebar.title("Fuel Efficiency prediction")

    # st.radio('Type of Cab you want to Book', options=['Mini', 'Sedan', 'XL', 'Premium', 'Rental'])
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Cars Fuel Efficiency Prediction App </h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    st.text("")
    

    uploadedFile = st.sidebar.file_uploader("Choose a file", type=['csv','xlsx'], accept_multiple_files=False, key="fileUploader")
    if uploadedFile is not None :
        try:

            startup = pd.read_csv(uploadedFile)
        except:
                try:
                    startup = pd.read_excel(uploadedFile)
                except:      
                    startup = pd.DataFrame()
        
        
    else:
        st.sidebar.warning("You need to upload a CSV or an Excel file.")
    
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <p style="color:white;text-align:center;">Add DataBase Credientials </p>
    </div>
    """
    st.sidebar.markdown(html_temp, unsafe_allow_html = True)
            
    user = st.sidebar.text_input("user", "Type Here")
    pw = st.sidebar.text_input("password", "Type Here")
    db = st.sidebar.text_input("database", "Type Here")
    
    result = ""
    
    if st.button("Predict"):
        result = predict_MPG(startup, user, pw, db)
        #st.dataframe(result) or
        #st.table(result.style.set_properties(**{'background-color': 'white','color': 'black'}))
                           
        import seaborn as sns
        cm = sns.light_palette("blue", as_cmap = True)
        # Apply the background_gradient to the Styler object
        styled_df = result.style.background_gradient(cmap=cm)

        # Set the precision for floating-point numbers
        styled_df = styled_df.format(precision=2)

        # Display the styled DataFrame in Streamlit
        st.table(styled_df)
        #st.table(result.style.background_gradient(cmap=cm).set_precision(2))

if __name__=='__main__':
    main()


