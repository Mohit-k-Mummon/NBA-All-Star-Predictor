import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

training_df = pd.read_csv("./nba_allstar_training_data.csv")

model_data = joblib.load('./All-Star Predictor.joblib')
model = model_data['model']


st.write("""
# NBA All-Star Prediction App
         
This app predicts whether an NBA Player will be an **All-Star**!
         
Click the Chevron Icon in the top left corner to adjust the NBA Players stats
         
Dataset used: https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats?select=Player+Per+Game.csv
""")

st.sidebar.header('Player Stats Parameters')

def user_input_features():
  pts_per_game = st.sidebar.slider('Points Per Game', 0, 50, 10)
  asts_per_game = st.sidebar.slider('Assists Per Game', 0, 30, 10)
  reb_per_game = st.sidebar.slider('Rebounds Per Game', 0, 30, 10)
  blk_per_game = st.sidebar.slider('Blocks Per Game', 0, 20, 2)
  stls_per_game = st.sidebar.slider('Steals Per Game', 0, 10, 1)
  tov_per_game = st.sidebar.slider('Turnovers Per Game', 0, 15, 2)
  data = {'ppg': pts_per_game,
          'apg': asts_per_game,
          'rpg': reb_per_game,
          'bpg': blk_per_game,
          'spg': stls_per_game,
          'tpg': tov_per_game}
  features = pd.DataFrame(data, index=[0])
  return features

df = user_input_features()

st.subheader('User Input Parameters')
st.write(df)

labels_data = {"Label": ["Not All-Star", "All-Star"]}
labels_df = pd.DataFrame(labels_data)

st.subheader('Class Labels and Index')
st.write(labels_df)

dfv = df.values[0]
features = [[dfv[2], dfv[1], dfv[4], dfv[3], dfv[5], dfv[0]]]
prediction = model.predict(features)
prediction_proba = model.predict_proba(features)

st.subheader('Prediction')
st.write("All-Star" if prediction == 1 else "Not All-Star")
st.write(prediction)

st.subheader('Prediction Confidence')
st.write(prediction_proba)


# Bar Chart
st.subheader("All-Star vs Not All-Star Counts (2020 - 2024)")
# Count values
counts = training_df['AllStar'].value_counts().sort_index()  # ensures 0, then 1
labels = ['Not All-Star', 'All-Star']
# Plot
fig1, ax1 = plt.subplots()
ax1.bar(labels, counts, color=['gray', 'blue'])
ax1.set_ylabel("Count")
ax1.set_title("All-Star Class Distribution")
st.pyplot(fig1)


# Scatter Plot
st.subheader("Scatter Plot: Points vs Assists")
# Split data by class
not_allstar = training_df[training_df['AllStar'] == 0]
allstar = training_df[training_df['AllStar'] == 1]
fig2, ax2 = plt.subplots()
ax2.scatter(not_allstar['pts_per_game'], not_allstar['ast_per_game'], color='gray', label='Not All-Star', alpha=0.6)
ax2.scatter(allstar['pts_per_game'], allstar['ast_per_game'], color='blue', label='All-Star', alpha=0.6)
ax2.set_xlabel('Points Per Game')
ax2.set_ylabel('Assists Per Game')
ax2.set_title('PTS vs AST by All-Star Status')
ax2.legend()
st.pyplot(fig2)



# Histogram
st.subheader("Histogram: Points Per Game")
fig3, ax3 = plt.subplots()
ax3.hist(training_df['pts_per_game'], bins=20, color='orange', edgecolor='black')
ax3.set_xlabel('Points Per Game')
ax3.set_ylabel('Frequency')
ax3.set_title('Distribution of Points Per Game')
st.pyplot(fig3)



# Feature Importance
# Get feature names and coefficients
st.subheader("Feature Importance")
feature_names = ['rpg', 'apg', 'spg', 'bpg', 'tpg', 'ppg']
coefficients = model.coef_[0]
# Plot
fig4, ax = plt.subplots()
ax.barh(feature_names, coefficients)
ax.set_title('Feature Importance (Logistic Coefficients)')
st.pyplot(fig4)
