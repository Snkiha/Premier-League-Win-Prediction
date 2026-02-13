# Premier League Win Predictor
- A machine learning project that uses the LightBGM, XGBoost, and Random Forest Classifier to predict the outcome of English Premier League matches (2022-2023). By analyzing historical match data and team form, this model estimates the probability of a home win, away win, or draw between both teams.

# Project Overview
- Predicting football results is notoriously difficult due to the "any given Sunday" nature of the sport. This project applies a binary and the aforementioned classifiers framework to quantify the influence of home advantage, recent form, and team strength on the final whistle.
## Key Features:
- Data Sourcing: Historical match data from the last 5+ seasons.

- Feature Engineering: Implementation of goal differences and rolling averages of Home and Away teams, expected goals, and rolling average of shot accuracy.

- Predictive Modeling: XGBoost, Random Forest and LightGBM calibrated for match-day probabilities.

# What is XGBoost
- XGBoost (eXtreme Gradient Boosting) is an advanced machine learning algorithm designed for efficiency, speed, and high performance.
- An optimized implementation of Gradient Boosting and is a type of ensemble learning method that combines multiple weak models to form a stronger one.

# What is LightGBM
- LightGBM (Light Gradient Boosting Machine) is an open-source, high-performance gradient boosting framework developed by Microsoft, specialized for fast, efficient, and distributed machine learning.
- Using tree-based algorithms with leaf-wise growth, it offers faster training, lower memory usage, and better accuracy for classification and regression tasks.