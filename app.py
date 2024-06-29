import streamlit as st
import pandas as pd
import numpy as np
import pickle as pkl

# Title and description of the web application
st.write('''
 # Iris Prediction app :bouquet:

    a simple application to predict **3 types of iris flowers** according to their *sepal and petal width and length*.
         ''')

# Sidebar for user inputs
st.sidebar.header('User input data')

def user_inputs():
    # Sliders for user inputs
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 6.5)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.2)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 4.5)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 1.7)

    # Constructing a dataframe with user inputs
    data = {'sepal_length' : sepal_length,
            'sepal_width' : sepal_width,
            'petal_length' : petal_length,
            'petal_width' : petal_width}
    
    features = pd.DataFrame(data, index = [0])
    return features

# Calling the function to get user inputs
iris_df = user_inputs()

# Displaying user inputs under 'User input' section
st.subheader('User input')
st.write(iris_df)

# Loading the pre-trained KNN model from a pickle file
load_knn = pkl.load(open('iris_classifier.pkl', 'rb'))

# Making prediction on user input data
pred = load_knn.predict(iris_df)
pred_prob = load_knn.predict_proba(iris_df)

# Displaying predicted Iris species
st.subheader('Prediction')
iris_genus = np.array(['Setosa', 'Versicolor', 'Virginica'])
st.write(iris_genus[pred])

# Displaying the probabilities for each Iris species
st.subheader('Prediction Probability')
st.write(pred_prob)