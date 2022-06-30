import streamlit as st

def app(df):
    # Setting the title to the home page contents.
    st.title("A Web App to Predict Early Diabetes")
    # brief description for the web app.
    st.markdown("""<p style='color:Purple;font-size:20px'>Diabetes is a chronic disease that occurs either when the pancreas does not produce enough insulin or when the body cannot effectively use the insulin it produces. Insulin is a hormone that regulates blood sugar.(SRC:WHO)
                This Web app will help you to predict whether a person has diabetes or is prone to get diabetes in future by analysing the values of several features using the Decision Tree Classifier.""", unsafe_allow_html = True) 

    st.header("Dig Deep and discover the data")

    with st.beta_expander("Dive into Full Dataset"):
        st.table(df)

    st.subheader("Columns Description:")
    col1, col2, col3 = st.beta_columns(3)

    # Adding a checkbox in the first column. Display the column names of 'df' on the click of checkbox
    with col1:
        if st.checkbox("View all column names"):
            st.table(list(df.columns))

    # Adding a checkbox in the second column. Display the column data-types of 'df' on the click of checkbox.
    with col2:
        if st.checkbox("View column data-types"):
            st.table(df.dtypes)

    # Adding a checkbox in the third column followed by a selectbox which accepts the column name whose data needs to be displayed.
    with col3:
        if st.checkbox("View column data"):
            column_data = st.selectbox('Select column', tuple(df.columns))
            st.write(df[column_data])

    if st.checkbox("Show data summary"):
        st.table(df.describe())
