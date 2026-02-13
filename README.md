# Premier League Win Predictor
- A machine learning project that uses the LightBGM, XGBoost, and Random Forest Classifier to predict the outcome of English Premier League matches (2022-2023). By analyzing historical match data and team form, this model estimates the probability of a home win, away win, or draw between both teams.

# Project Overview
- Predicting football results is notoriously difficult due to the "any given Sunday" nature of the sport. This project applies a binary and the aforementioned classifiers framework to quantify the influence of home advantage, recent form, and team strength on the final whistle.
## Key Features:
- Data Sourcing: Historical match data from the last 5+ seasons.

- Feature Engineering: Implementation of goal differences and rolling averages of Home and Away teams, expected goals, and rolling average of shot accuracy.

- Predictive Modeling: XGBoost, Random Forest and LightGBM calibrated for match-day probabilities.
