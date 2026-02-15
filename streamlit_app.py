import joblib
import numpy as np
import pandas as pd
import streamlit as st

model = joblib.load("full_pipeline.pkl")

with open("feature_columns.pkl", "rb") as f:
    feature_cols = joblib.load(f)

st.set_page_config(page_title = "Football Match Predictor", layout = "wide")
st.title("EPL Match Predictor")
st.write("Predicts probability of Home Team Winning (Data is currently trained over 2022-2023 seasons)")

st.divider()

# User Input
col1, col2 = st.columns(2)

with col1:
    home_team = st.text_input("Home Team")
    home_rolling_avg_xG = st.number_input("Home Rolling Average Expected Goals", min_value = 0.0)
    home_goal_chances = st.number_input("Home Goal Conversion Rate", min_value = 0.0)

with col2:
    away_team = st.text_input("Away Team")
    away_rolling_avg_xG = st.number_input("Away Rolling Average Expected Goals", min_value = 0.0)
    away_goal_chances = st.number_input("Away Goal Conversion Rate", min_value = 0.0)

avg_shot_diff = st.number_input("Average Shot Accuracy Difference", value = 0.0)
avg_goal_diff = st.number_input("Average Goal Difference", value = 0.0)
avg_possess_diff = st.number_input("Average Possession Difference", value = 0.0)

st.divider()

# Predict Button
if st.button("Predict"):
    
    input_dict = {
        "Home Team": home_team,
        "Away Team": away_team,
        "avg_shot_diff": avg_shot_diff,
        "avg_goal_diff": avg_goal_diff,
        "avg_possess_diff": avg_possess_diff,
        "home_goal_chances": home_goal_chances,
        "away_goal_chances": away_goal_chances,
        "home_rolling_avg_xG": home_rolling_avg_xG,
        "away_rolling_avg_xG": away_rolling_avg_xG
    }
    input_df = pd.DataFrame([input_dict])
    
    # Reindex to match training columns
    input_df = input_df.reindex(columns=feature_cols, fill_value=0)
    prob = model.predict_proba(input_df)[:, 1][0]
    
    st.subheader("Prediction Results")
    st.metric("Home Win Probability", f"{prob:.2%}")
    
    if prob > 0.6:
        st.success("Strong Home Win Signal")
    elif prob < 0.4:
        st.warning("Away Side Favored")
    else:
        st.info("Close Match")