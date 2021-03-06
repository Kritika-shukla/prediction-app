
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
 

# Define a function 'app()' which accepts 'census_df' as an input.
def app(df):
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Visualise Data")
    st.subheader("Visualisation Selector")

    # Add a multiselect in the sidebar with label 'Select the Charts/Plots:'
    # Store the current value of this widget in a variable 'plot_list'.
    plot_list = st.sidebar.multiselect("Select the Charts/Plots:", ('Correlation Heatmap', 'Line Chart', 'Area Chart', 'Count Plot','Pie Chart', 'Box Plot','Pair Plots'))

    # Displaying Line Chart
    if 'Line Chart' in plot_list:
        st.subheader("Line Chart")
        st.line_chart(df)

   # Displaying area chart    
    if 'Area Chart' in plot_list:
        st.subheader("Area Chart")
        st.area_chart(df)
       
   # Displaying Count Plot 
    if 'Count Plot' in plot_list:
        st.subheader("Count plot")
        sns.countplot(x = 'Outcome', data = df)
        st.pyplot()
        
    # Displaying Correlation Heatmap
    if 'Correlation Heatmap' in plot_list:
        st.subheader("Correlation Heatmap")
        plt.figure(figsize = (10, 6))
        hmap = sns.heatmap(df.iloc[:, 1:].corr(), annot = True,cmap = 'YlGnBu') # an object of seaborn axis in 'hmap' variable
        bottom, top = hmap.get_ylim() # Getting the top and bottom margin limits.
        hmap.set_ylim(bottom + 0.5, top - 0.5) # Increasing the bottom and decreasing the top margins respectively.
        st.pyplot()

    # Displaying pie plot
    if 'Pie Chart' in plot_list:
        st.subheader("Pie Chart")
        pie_data = df['Outcome'].value_counts()
        plt.figure(dpi =96)
        plt.pie(pie_data, labels = pie_data.index, autopct = '%1.2f%%', startangle = 30, explode = [0,0.2])
        st.pyplot()

    # Displaying box plot
    if 'Box Plot' in plot_list:
        st.subheader("Box Plot")
        column = st.sidebar.selectbox("Select the column for boxplot",('BMI','Diabetes','Glucose','BP','Age'))
        sns.boxplot(df[column])
        st.pyplot()
        
    # Displaying Pair plot    
    if 'Pair Plot' in plot_list:
        st.subheader("Pair Plots")
        plt.figure(figsize = (15, 15))
        sns.pairplot(df)
        st.pyplot()

    features_list = st.sidebar.multiselect("Scatterplot : Select value for x- axis ",('BMI','Glucose','BP','Age'))

    for feature in features_list:
        st.subheader(f"Scatter plot between {feature} and Outcome")
        plt.figure(figsize = (12, 6))
        sns.scatterplot(x = feature, y = 'Outcome', data = df)
        st.pyplot()
