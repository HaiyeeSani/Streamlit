pip install streamlit
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Load the datasets and preprocess
VE = pd.read_excel('entreprenure.xlsx', sheet_name='Vietnam', usecols=['EI1', 'EI2', 'EI3', 'EI4', 'EI5', 'EI6'])
VA = pd.read_excel('entreprenure.xlsx', sheet_name='Vietnam', usecols=['ATE1', 'ATE2', 'ATE3', 'ATE4', 'ATE5'])
VP = pd.read_excel('entreprenure.xlsx', sheet_name='Vietnam', usecols=['PBC1', 'PBC2', 'PBC3', 'PBC4', 'PBC5', 'PBC6'])

X = pd.concat([VA, VP], axis=1)
VE['Combined'] = VE.mean(axis=1)
y = VE['Combined']

# Train the DecisionTreeRegressor model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# Streamlit app
st.title('Entrepreneurship Intention Predictor')

ATE1 = st.number_input('ATE1', min_value=0, max_value=5, value=0)
ATE2 = st.number_input('ATE2', min_value=0, max_value=5, value=0)
ATE3 = st.number_input('ATE3', min_value=0, max_value=5, value=0)
ATE4 = st.number_input('ATE4', min_value=0, max_value=5, value=0)
ATE5 = st.number_input('ATE5', min_value=0, max_value=5, value=0)
PBC1 = st.number_input('PBC1', min_value=0, max_value=5, value=0)
PBC2 = st.number_input('PBC2', min_value=0, max_value=5, value=0)
PBC3 = st.number_input('PBC3', min_value=0, max_value=5, value=0)
PBC4 = st.number_input('PBC4', min_value=0, max_value=5, value=0)
PBC5 = st.number_input('PBC5', min_value=0, max_value=5, value=0)
PBC6 = st.number_input('PBC6', min_value=0, max_value=5, value=0)

if st.button('Predict'):
    inputs = [[ATE1, ATE2, ATE3, ATE4, ATE5, PBC1, PBC2, PBC3, PBC4, PBC5, PBC6]]
    prediction = model.predict(inputs)
    st.success(f'Predicted Entrepreneurship Intention: {prediction[0]}')

