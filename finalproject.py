import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

# importing data
df = pd.read_csv("py4ai-score.csv")
print(df)

# cleaning data
df['BONUS'].fillna(0, inplace=True)
df['S1'].fillna(0, inplace=True)
df['S2'].fillna(0, inplace=True)
df['S3'].fillna(0, inplace=True)
df['S4'].fillna(0, inplace=True)
df['S5'].fillna(0, inplace=True)
df['S6'].fillna(0, inplace=True)
df['S7'].fillna(0, inplace=True)
df['S8'].fillna(0, inplace=True)
df['S9'].fillna(0, inplace=True)
df['S10'].fillna(0, inplace=True)
df['REG-MC4AI'].fillna('N', inplace=True)
print(df.isnull().sum())

# data analysis
# pie chart(s) thể hiện tỷ lệ học sinh
nam_nu = px.pie(df, names='GENDER')
nam_nu.show()
