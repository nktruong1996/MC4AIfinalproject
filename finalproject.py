import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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
nam_nu = px.pie(df, names='GENDER', color_discrete_sequence=['#1f77b4', '#d62728'])        # tỷ lệ nam nữ

def createpc(row):
    return row['PYTHON-CLASS'][0:3]

df['PC'] = df.apply(createpc, axis=1)

pc = px.pie(df, names='PC', color_discrete_sequence=['#1f77b4', '#d62728'])                # tỷ lệ lớp 114 và 115

def creategrade(row):
    return row['CLASS'][0:2]

df['GRADE'] = df.apply(creategrade, axis=1)

khoilop = px.pie(df, names='GRADE', color_discrete_sequence=['#1f77b4', '#d62728', '#bcbd22'])  # tỷ lệ học sinh giữa các khối lớp

def createperiod(row):
    if row['PYTHON-CLASS'][-1] == 'C': return 'C'
    elif row['PYTHON-CLASS'][-1] == 'S': return 'S'

df['PERIOD'] = df.apply(createperiod, axis=1)
print(df.head())

sc = px.pie(df, names='PERIOD', color_discrete_sequence=['#1f77b4', '#d62728']) # tỷ lệ học sinh sáng và chiều

def create_class_group(row):
  if row['CLASS'][2:6] == 'CTIN': return 'Chuyên Tin'
  elif row['CLASS'][2:6] == 'CTRN': return 'Trung Nhật'
  elif row['CLASS'][2:4] == 'CT': return 'Chuyên Toán'
  elif row['CLASS'][2:4] == 'CL': return 'Chuyên Lý'
  elif row['CLASS'][2:4] == 'CV': return 'Chuyên Văn'
  elif row['CLASS'][2:4] == 'CH': return 'Chuyên Hóa'
  elif row['CLASS'][2:4] == 'CA': return 'Chuyên Anh'
  elif row['CLASS'][2:4] == 'SN': return 'Tích Hợp/Song Ngữ'
  elif row['CLASS'][2:4] == 'TH': return 'Tích Hợp/Song Ngữ'
  elif row['CLASS'][2:5] == 'CSD': return 'Sử Địa'
  else: return 'Khác'

df['CLASS-GROUP'] = df.apply(create_class_group, axis=1)

cg = px.pie(df, names='CLASS-GROUP', color_discrete_sequence= ['#1f77b4',           # tỷ lệ học sinh các loại lớp phổ thông
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',])

# box charts so sánh điểm từng session và GPA theo class groups
box1=px.box(df, x = 'CLASS-GROUP', y='S1', color_discrete_sequence= ['#1f77b4'])
box2=px.box(df, x = 'CLASS-GROUP', y='S2', color_discrete_sequence= ['#1f77b4'])
box3=px.box(df, x = 'CLASS-GROUP', y='S3', color_discrete_sequence= ['#1f77b4'])
box4=px.box(df, x = 'CLASS-GROUP', y='S4', color_discrete_sequence= ['#1f77b4'])
box5=px.box(df, x = 'CLASS-GROUP', y='S5', color_discrete_sequence= ['#1f77b4'])
box6=px.box(df, x = 'CLASS-GROUP', y='S6', color_discrete_sequence= ['#1f77b4'])
box7=px.box(df, x = 'CLASS-GROUP', y='S7', color_discrete_sequence= ['#1f77b4'])
box8=px.box(df, x = 'CLASS-GROUP', y='S8', color_discrete_sequence= ['#1f77b4'])
box9=px.box(df, x = 'CLASS-GROUP', y='S9', color_discrete_sequence= ['#1f77b4'])
box10=px.box(df, x = 'CLASS-GROUP', y='S10', color_discrete_sequence= ['#1f77b4'])
boxGPA=px.box(df, x = 'CLASS-GROUP', y='GPA', color_discrete_sequence= ['#1f77b4'])

def createban(row):
    if row['CLASS-GROUP'][7:] == 'Tin' or row['CLASS-GROUP'][7:] == 'Toán' or row['CLASS-GROUP'][7:] == 'Lý' or row['CLASS-GROUP'][6:-1] == 'Hóa': return 'TN'
    elif row['CLASS-GROUP'][7:] == 'Văn' or row['CLASS-GROUP'][:2] == 'Sử': return 'XH'
    else: return 'K'

df['BAN'] = df.apply(createban, axis=1)

boxBAN=px.box(df, x = 'BAN', y='GPA', color_discrete_sequence= ['#1f77b4']) # box chart thể hiện tương quan điểm theo ban

def qual(row):
    if row['GPA']>=6.5: return 1
    else: return 0

df['PASS'] = df.apply(qual, axis=1)

hist_cg=px.histogram(df, x='CLASS-GROUP',color='PASS', color_discrete_sequence= ['#1f77b4','#d62728']) # histogram thể hiện số lượng học sinh đậu/rớt theo class groups

cont=px.histogram(df, x='CLASS-GROUP',color='REG-MC4AI', color_discrete_sequence= ['#1f77b4','#d62728']) # histogram thể hiện số lượng học sinh đăng ký học tiếp theo class groups

# dùng KMeans để phân loại học sinh
X = df[['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10']].values

# dùng phương pháp elbow để xác định k hợp lý
K = [k for k in range(1, 10)]
W = []

for k in K:
  kmeans = KMeans(n_clusters=k, n_init='auto')
  kmeans.fit(X)
  w = 0
  for i in range(k):
    Xi = X[kmeans.labels_ == i]
    wi = sum([np.linalg.norm(x-kmeans.cluster_centers_[i]) for x in Xi])
    w += wi
  W.append(w)
plot0=plt.plot(K, W, marker = 'o')
plt.savefig('general_classification.png', bbox_inches='tight')
# dựa vào đồ thị bên trên, k=3,5,6 là hợp lý, chọn k=5
# fit dữ liệu vào thuật toán KMeans với k=5
kmeans = KMeans(n_clusters=5, n_init='auto')
kmeans.fit(X)
classes = np.array(kmeans.labels_)
print(classes)

df['PL'] = classes
classification=px.pie(df, names='PL', color_discrete_sequence=['#1f77b4'])

X_ = df[['GPA']].values
K = [k for k in range(1, 10)]
W = []

for k in K:
  kmeans = KMeans(n_clusters=k, n_init='auto')
  kmeans.fit(X_)
  w = 0
  for i in range(k):
    Xi = X_[kmeans.labels_ == i]
    wi = sum([np.linalg.norm(x-kmeans.cluster_centers_[i]) for x in Xi])
    w += wi
  W.append(w)

plot1=plt.plot(K, W, marker = 'o')
plt.savefig('general_gpa_classification.png', bbox_inches='tight')
# dựa trên fig này thì ta thấy rằng phân loại số lượng cluster bằng mấy thì phân loại theo tất cả các cột điểm cũng sẽ tương đối giống với phân loại theo GPA

X0 = df[kmeans.labels_ == 0]['GPA'].mean()
X1 = df[kmeans.labels_ == 1]['GPA'].mean()
X2 = df[kmeans.labels_ == 2]['GPA'].mean()
X3 = df[kmeans.labels_ == 3]['GPA'].mean()
X4 = df[kmeans.labels_ == 4]['GPA'].mean()
print(X0,X1,X2,X3,X4)

#
