import streamlit as st
import numpy as np
import pandas as pd

# Configure your home page by setting its title and icon that will be displayed in a browser tab.
st.set_page_config(page_title = 'Early Diabetes Prediction Web App',
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


# Create the Page Navigator for 'Home', 'Predict Diabetes' and 'Visualise Decision Tree' web pages in 'main.py'
# Import the 'predict' 'home', 'plots' Python files
import predict
import home
import plot

# Adding a navigation in the sidebar using radio buttons
# Create a dictionary.
pages_dict = {"Home": home, 
              "Predict Diabetes": predict, 
              "Visualise Decision Tree": plot}

# Add radio buttons in the sidebar for navigation and call the respective pages based on user selection.
st.sidebar.title('Navigation')
user_choice = st.sidebar.radio("Go to", tuple(pages_dict.keys()))
selected_page = pages_dict[user_choice]
selected_page.app(df) 