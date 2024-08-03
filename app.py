import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
model = pickle.load(open('D:/project/ipl_score_predictor/first-innings-score-lr-model.pkl', 'rb'))

# Load feature names
with open('D:/project/ipl_score_predictor/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# List of teams
teams = [
    'Chennai Super Kings', 'Delhi Daredevils', 'Kings XI Punjab',
    'Kolkata Knight Riders', 'Mumbai Indians', 'Rajasthan Royals',
    'Royal Challengers Bangalore', 'Sunrisers Hyderabad'
]

# Page title
st.title('IPL First Innings Score Prediction')

# User inputs
bat_team = st.selectbox('Select the batting team', sorted(teams))
bowl_team = st.selectbox('Select the bowling team', sorted(teams))
overs = st.number_input('Enter the number of overs bowled', min_value=5.0, max_value=19.4, step=0.1)
runs = st.number_input('Enter the number of runs scored', min_value=0,max_value=266)
wickets = st.number_input('Enter the number of wickets lost', min_value=0, max_value=10)
runs_last_5 = st.number_input('Enter the runs scored in the last 5 overs', min_value=0,max_value=110)
#wickets_last_5 = st.number_input('Enter the wickets lost in the last 5 overs', min_value=0)

# Convert categorical variables using OneHotEncoding
input_data = {
    'overs': overs,
    'runs': runs,
    'wickets': wickets,
    'runs_last_5': runs_last_5,
    #'wickets_last_5': wickets_last_5
}

for team in teams:
    input_data[f'bat_team_{team}'] = 1 if team == bat_team else 0
    input_data[f'bowl_team_{team}'] = 1 if team == bowl_team else 0

input_df = pd.DataFrame([input_data])

# Ensure the input dataframe has the same columns as the training data
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Predict button
if st.button('Predict Score'):
    prediction = model.predict(input_df)
    st.write(f'The predicted first innings score is: {int(prediction[0])}')
