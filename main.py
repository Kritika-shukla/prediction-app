import streamlit as st
import pandas as pd
import numpy as np

# Configuring the home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title = 'A Web App to Predict Early Diabetes',
                    page_icon = 'random',
                    layout = 'wide',
                    initial_sidebar_state = 'auto'
                    )

# Loading the dataset.
@st.cache()
def data_load():
    # Load the Diabetes dataset into DataFrame.

    df = pd.read_csv('diabetes.csv')
    df.head()

    # Rename the column names in the DataFrame.
    df.rename(columns = {"BloodPressure": "BP",}, inplace = True)
    df.rename(columns = {"SkinThickness": "Skin_Thickness",}, inplace = True)
    df.rename(columns = {"DiabetesPedigreeFunction": "Pedigree",}, inplace = True)

    df.head() 

    return df

df = data_load()

# Importing the 'predict' 'home', 'plots' Python files
import predict
import home
import plot

# Adding a navigation in the sidebar using radio buttons
# Creating a dictionary.
multipage_dict = {"Exploratory Data Analysis": home, 
              "Prediction": predict, 
              "Data Visualisation": plot}

# Adding radio buttons in the sidebar for navigation and call the respective pages based on user choice.
st.sidebar.title('Navigation')
choice = st.sidebar.radio("Go to", tuple(multipage_dict.keys()))
opted_page = multipage_dict[choice]
opted_page.app(df) 
