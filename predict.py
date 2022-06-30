#importing modules
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

# ML classifier Python modules
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

@st.cache(suppress_st_warning=True)
def log_pred(df, glucose, bp, insulin, bmi, pedigree, age):
    # Split the train and test dataset. 
    feat_cols = list(df.columns)

    # Removing the 'Pregnancies', Skin_Thickness' columns and the 'target' column from the feature columns
    feat_cols.remove('Skin_Thickness')
    feat_cols.remove('Pregnancies')
    feat_cols.remove('Outcome')

    X = df[ feat_cols]
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    log_reg = LogisticRegression(C = 1, max_iter = 10)
    log_reg.fit(X_train, y_train)
    
    # Predicting diabetes using the 'predict()' function.
    prediction = log_reg.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(log_reg.score(X_test, y_test)*100,3)

    return prediction, score

def rand_for_pred(df, glucose, bp, insulin, bmi, pedigree, age):    
    feat_cols = list(df.columns)
    # Remove the 'Pregnancies', 'Skin_Thickness' columns and the 'target' column from the feature columns
    feat_cols.remove('Pregnancies')
    feat_cols.remove('Skin_Thickness')
    feat_cols.remove('Outcome')
    X = df[feat_cols]
    y = df['Outcome']
    # Split the train and test dataset. 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    # Training
    rf_clf = RandomForestClassifier(n_estimators = 100, max_depth = 1, n_jobs = -1)
    rf_clf.fit(X_train,y_train)
    
    # Predict diabetes using the 'predict()' function.
    prediction = rf_clf.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(rf_clf.score(X_test, y_test)*100,3)

    return prediction, score

def d_tree_pred(df, glucose, bp, insulin, bmi, pedigree, age):
    # Split the train and test dataset. 
    feat_cols = list(df.columns)

    # Remove the 'Pregnancies', Skin_Thickness' columns and the 'target' column from the feature columns
    feat_cols.remove('Skin_Thickness')
    feat_cols.remove('Pregnancies')
    feat_cols.remove('Outcome')

    X = df[feat_cols]
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

    dtree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)
    dtree_clf.fit(X_train, y_train) 
    y_train_pred = dtree_clf.predict(X_train)
    y_test_pred = dtree_clf.predict(X_test)
    # Predict diabetes using the 'predict()' function.
    prediction = dtree_clf.predict([[glucose, bp, insulin, bmi, pedigree, age]])
    prediction = prediction[0]

    score = round(metrics.accuracy_score(y_train, y_train_pred) * 100, 3)

    return prediction, score

    # Creating the user defined 'app()' function.
def app(df):
    st.markdown("<p style='color:Purple;font-size:25px'>This Web app uses <b> Classifiers </b> for the Early Prediction of Diabetes.", unsafe_allow_html = True) 
    st.subheader("Select Values:")

    glucose = st.slider("Glucose", int(df["Glucose"].min()), int(df["Glucose"].max()))
    bp = st.slider("Blood Pressure", int(df["BP"].min()), int(df["BP"].max()))
    insulin = st.slider("Insulin", int(df["Insulin"].min()), int(df["Insulin"].max()))
    bmi = st.slider("BMI", float(df["BMI"].min()), float(df["BMI"].max()))
    pedigree = st.slider("Pedigree Function", float(df["Pedigree"].min()), float(df["Pedigree"].max()))
    age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()))


    st.subheader("Select the model")

    # Add a single select drop down menu with label 'Select the Classifier'
    predictor = st.selectbox("Select the Classifier",('Random Forest Classifier', 'Logistic Regression','Decision Tree Classifier'))

    if predictor == 'Random Forest Classifier':
        if st.button("Predict"): 
     
            prediction, score = rand_for_pred(df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Prediction results for Random Forest Classifier:")
            if prediction == 1:
                st.info("The person either has diabetes or prone to get diabetes")
            else:
                st.info("The person is free from diabetes")
            st.write(f"The accuracy score of this model is, {score} %")


    elif predictor == 'Logistic Regression':
        if st.button("Predict"):
            prediction, score = log_pred(df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Decision Tree Prediction results with Logistic Regression:")
            if prediction == 1:
                st.info("This person may have diabetes or is prone to it.")
            else:
                st.info("This person is free from diabetes")
            st.write(f"The accuracy score of this model is, {score} %")


    if predictor == 'Decision Tree Classifier':
        if st.button("Predict"):            
            prediction, score = d_tree_pred(df, glucose, bp, insulin, bmi, pedigree, age)
            st.subheader("Decision Tree Prediction results:")
            if prediction == 1:
                st.info("This person may have diabetes or is prone to it.")
            else:
                st.info("This person is free from diabetes")
            st.write(f"The accuracy score of this model is, {score} %")

