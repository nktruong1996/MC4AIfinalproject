import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Input, Flatten
from tensorflow.random import set_seed
from tensorflow.keras.backend import clear_session
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


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
    return int(row['CLASS'][0:2])

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
# dựa vào đồ thị bên trên, k=2,3,4,5 là hợp lý

#fit dữ liệu vào thuật toán KMeans với k=2
kmeans2 = KMeans(n_clusters=2, n_init='auto')
kmeans2.fit(X)
classes = np.array(kmeans2.labels_)
X20 = df[kmeans2.labels_ == 0]['GPA'].mean()
X21 = df[kmeans2.labels_ == 1]['GPA'].mean()
groups2 = go.Figure(data=[go.Scatter3d(x=df[kmeans2.labels_ == 0]['S6'], y=df[kmeans2.labels_ == 0]['S10'], z=df[kmeans2.labels_ == 0]['GPA'], mode='markers', marker = {'color' : 'blue'}),
    go.Scatter3d(x=df[kmeans2.labels_ == 1]['S6'], y=df[kmeans2.labels_ == 1]['S10'], z=df[kmeans2.labels_ == 1]['GPA'], mode='markers', marker = {'color' : 'red'})])

# fit dữ liệu vào thuật toán KMeans với k=3
kmeans3 = KMeans(n_clusters=3, n_init='auto')
kmeans3.fit(X)
classes = np.array(kmeans3.labels_)
X30 = df[kmeans3.labels_ == 0]['GPA'].mean()
X31 = df[kmeans3.labels_ == 1]['GPA'].mean()
X32 = df[kmeans3.labels_ == 2]['GPA'].mean()
groups3 = go.Figure(data=[go.Scatter3d(x=df[kmeans3.labels_ == 0]['S6'], y=df[kmeans3.labels_ == 0]['S10'], z=df[kmeans3.labels_ == 0]['GPA'], mode='markers', marker = {'color' : 'blue'}),
    go.Scatter3d(x=df[kmeans3.labels_ == 1]['S6'], y=df[kmeans3.labels_ == 1]['S10'], z=df[kmeans3.labels_ == 1]['GPA'], mode='markers', marker = {'color' : 'red'}),
    go.Scatter3d(x=df[kmeans3.labels_ == 2]['S6'], y=df[kmeans3.labels_ == 2]['S10'], z=df[kmeans3.labels_ == 2]['GPA'], mode='markers', marker = {'color' : 'green'})])

# fit dữ liệu vào thuật toán KMeans với k=4
kmeans4 = KMeans(n_clusters=4, n_init='auto')
kmeans4.fit(X)
classes = np.array(kmeans4.labels_)
X40 = df[kmeans4.labels_ == 0]['GPA'].mean()
X41 = df[kmeans4.labels_ == 1]['GPA'].mean()
X42 = df[kmeans4.labels_ == 2]['GPA'].mean()
X43 = df[kmeans4.labels_ == 3]['GPA'].mean()
groups4 = go.Figure(data=[go.Scatter3d(x=df[kmeans4.labels_ == 0]['S6'], y=df[kmeans4.labels_ == 0]['S10'], z=df[kmeans4.labels_ == 0]['GPA'], mode='markers', marker = {'color' : 'blue'}),
    go.Scatter3d(x=df[kmeans4.labels_ == 1]['S6'], y=df[kmeans4.labels_ == 1]['S10'], z=df[kmeans4.labels_ == 1]['GPA'], mode='markers', marker = {'color' : 'red'}),
    go.Scatter3d(x=df[kmeans4.labels_ == 2]['S6'], y=df[kmeans4.labels_ == 2]['S10'], z=df[kmeans4.labels_ == 2]['GPA'], mode='markers', marker = {'color' : 'green'}),
    go.Scatter3d(x=df[kmeans4.labels_ == 3]['S6'], y=df[kmeans4.labels_ == 3]['S10'], z=df[kmeans4.labels_ == 3]['GPA'], mode='markers', marker = {'color' : 'yellow'})])

# fit dữ liệu vào thuật toán KMeans với k=5
kmeans5 = KMeans(n_clusters=5, n_init='auto')
kmeans5.fit(X)
classes = np.array(kmeans5.labels_)
X50 = df[kmeans5.labels_ == 0]['GPA'].mean()
X51 = df[kmeans5.labels_ == 1]['GPA'].mean()
X52 = df[kmeans5.labels_ == 2]['GPA'].mean()
X53 = df[kmeans5.labels_ == 3]['GPA'].mean()
X54 = df[kmeans5.labels_ == 4]['GPA'].mean()
groups5 = go.Figure(data=[go.Scatter3d(x=df[kmeans5.labels_ == 0]['S6'], y=df[kmeans5.labels_ == 0]['S10'], z=df[kmeans5.labels_ == 0]['GPA'], mode='markers', marker = {'color' : 'blue'}),
    go.Scatter3d(x=df[kmeans5.labels_ == 1]['S6'], y=df[kmeans5.labels_ == 1]['S10'], z=df[kmeans5.labels_ == 1]['GPA'], mode='markers', marker = {'color' : 'red'}),
    go.Scatter3d(x=df[kmeans5.labels_ == 2]['S6'], y=df[kmeans5.labels_ == 2]['S10'], z=df[kmeans5.labels_ == 2]['GPA'], mode='markers', marker = {'color' : 'green'}),
    go.Scatter3d(x=df[kmeans5.labels_ == 3]['S6'], y=df[kmeans5.labels_ == 3]['S10'], z=df[kmeans5.labels_ == 3]['GPA'], mode='markers', marker = {'color' : 'yellow'}),
    go.Scatter3d(x=df[kmeans5.labels_ == 4]['S6'], y=df[kmeans5.labels_ == 4]['S10'], z=df[kmeans5.labels_ == 4]['GPA'], mode='markers', marker = {'color' : 'purple'})])

# multiclass regression: S-AVG + S6 -> S10
df['S-AVG'] = (df['S1'] + df['S2'] + df['S3'] + df['S4'] + df['S5'] + df['S7'] + df['S8'] + df['S9']) / 8
avg6to10 = go.Figure(data=[go.Scatter3d(x=df['S-AVG'], y=df['S6'], z=df['S10'], mode='markers', marker = {'color' : 'blue'})])

def softmax(z):         # hàm softmax
    e = np.exp(z)
    return e/e.sum()

# data prep for given labels
# X = df[['S-AVG','S6']].values

# y = df['S10'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=11, random_state=42)
# y_train_ohe = to_categorical(y_train, num_classes=11)
# y_test_ohe = to_categorical(y_test, num_classes=11)

# print(y_train.shape, y_train_ohe.shape, y_test.shape, y_test_ohe.shape)

# clear_session()
# set_seed(42)
# np.random.seed(42)

# model = Sequential()
# model.add(Input(shape=X_train.shape[1:]))
# model.add(Flatten())
# model.add(Dense(11, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
# model.summary()

# history = model.fit(X_train, y_train_ohe, epochs = 4500, verbose=1)

# plt.figure(figsize=(10,6))
# plt.title('Learning Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss | Accuracy')
# plt.legend(['Loss', 'Accuracy'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['accuracy'])
# plt.savefig('11_class_classification.png', bbox_inches='tight')

# # data prep for 3-class classification
# def createclass(row):
#     if row['S10'] >= 8: return 2            # 2: trên 8
#     elif row['S10'] >=5: return 1           # 1: từ 5 tới 7
#     else: return 0                          # 0: còn lại

# df['XL3'] = df.apply(createclass, axis=1)

# X = df[['S-AVG','S6']].values

# y = df['XL3'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=11, random_state=42)
# y_train_ohe = to_categorical(y_train, num_classes=3)
# y_test_ohe = to_categorical(y_test, num_classes=3)

# print(y_train.shape, y_train_ohe.shape, y_test.shape, y_test_ohe.shape)

# clear_session()
# set_seed(42)
# np.random.seed(42)

# model = Sequential()
# model.add(Input(shape=X_train.shape[1:]))
# model.add(Flatten())
# model.add(Dense(3, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
# model.summary()

# history = model.fit(X_train, y_train_ohe, epochs = 4500, verbose=1)

# plt.figure(figsize=(10,6))
# plt.title('Learning Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss | Accuracy')
# plt.legend(['Loss', 'Accuracy'])
# plt.plot(history.history['loss'])
# plt.plot(history.history['accuracy'])
# plt.savefig('3_class_classification.png', bbox_inches='tight')

# data prep for 2-class classification
def createclass2(row):
    if row['S10'] >= 6: return 1        # 1: từ 6 trở lên
    else: return 0                      # 0: còn lại

df['XL2'] = df.apply(createclass2, axis=1)
X = df[['S-AVG','S6']].values

y = df['XL2'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=15, random_state=41)
y_train_ohe = to_categorical(y_train, num_classes=2)
y_test_ohe = to_categorical(y_test, num_classes=2)

print(y_train.shape, y_train_ohe.shape, y_test.shape, y_test_ohe.shape)

clear_session()
set_seed(41)
np.random.seed(41)

model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics='accuracy')
model.summary()

history = model.fit(X_train, y_train_ohe, epochs = 4500, verbose=1)

twoclass = plt.figure(figsize=(10,6))
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss | Accuracy')
plt.legend(['Loss', 'Accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.savefig('2_class_classification.png', bbox_inches='tight')

y_test_pred = model.predict(X_test).argmax(axis=1)

cm = confusion_matrix(y_test, y_test_pred)
labels = ['Còn lại', 'Trên 5']
plt.figure(figsize=(3,3))
sns.heatmap(cm, annot=True, cmap='Blues', yticklabels=labels, xticklabels=labels)
plt.savefig('confusion_2_class.png', bbox_inches='tight')

# midterm + final -> GPA
midfintoGPA = go.Figure(data=[go.Scatter3d(x=df['S6'], y=df['S10'], z=df['GPA'], mode='markers', marker = {'color' : 'blue'})])


# linear regression cho trường hợp này
X = df[['S6', 'S10']].values
y = df['GPA'].values

# tạo và fit model với data
midfin = LinearRegression()
midfin.fit(X,y)

# evaluate
print(midfin.score(X, y))

# regression plane
x_plane = np.linspace(0,11,2)
y_plane = np.linspace(0,11,2)

xx, yy = np.meshgrid(x_plane, y_plane)
xy = np.c_[xx.ravel(), yy.ravel()]
z = midfin.predict(xy)
z = z.reshape(xx.shape)

midfin_plane = go.Figure(data=[go.Scatter3d(x=df['S6'], y=df['S10'], z=df['GPA'], mode='markers', marker = {'color' : 'green'}),
                      go.Surface(x=x_plane, y=y_plane, z=z)])

# midterm + homework avg -> pass/fail
X = df[['S6','S-AVG']].values
y = df['PASS']
print(X[y==0][:,0])

plt.figure(figsize=(5,5))
plt.scatter(X[y==0][:,0], X[y==0][:,1], label = 'failed')
plt.scatter(X[y==1][:,0], X[y==1][:,1], label = 'passed')
plt.legend()

# logistic regression cho trường hợp này
midavg = LogisticRegression()
midavg.fit(X, y)

weights = midavg.coef_[0]
bias = midavg.intercept_[0]
weights, bias
w1, w2 = weights

midavg_plot = plt.figure(figsize=(5,5))
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('S6')
plt.ylabel('S-AVG')

x1 = np.array([0,10])
x2 = (-w1 * x1 - bias) / w2
plt.plot(x1, x2)

# streamlit demo
st.title('Final Project - Phân tích bảng điểm lớp Python4AI')
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Danh sách", "Biểu đồ", "Phân nhóm", "Phân loại", "Dự đoán"])
with tab1:
    st.header('Tùy chọn hiển thị')
    st.subheader('Giới tính')
    genders = []
    grades = []
    periods = []
    phong = []
    lopchuyen = []

    gender_choices = st.columns(2)
    with gender_choices[0]:
        nam = st.checkbox('Nam')
    with gender_choices[1]:
        nu = st.checkbox('Nữ')
    if nam: genders.append('M')
    if nu: genders.append('F')
    gender_check = np.array([1 if df.loc[i, 'GENDER'] in genders else 0 for i in df.index])

    st.subheader('Khối lớp')
    grade_choices = st.columns(3)
    with grade_choices[0]:
        khoi10 = st.checkbox('10')
    with grade_choices[1]:
        khoi11 = st.checkbox('11')
    with grade_choices[2]:
        khoi12 = st.checkbox('12')
    if khoi10: grades.append(10)
    if khoi11: grades.append(11)
    if khoi12: grades.append(12)
    grade_check = np.array([1 if df.loc[i, 'GRADE'] in grades else 0 for i in df.index])

    st.subheader('Buổi')
    period_choices = st.columns(2)
    with period_choices[0]:
        sang = st.checkbox('Sáng')
    with period_choices[1]:
        chieu = st.checkbox('Chiều')
    if sang: periods.append('S')
    if chieu: periods.append('C')
    period_check = np.array([1 if df.loc[i, 'PERIOD'] in periods else 0 for i in df.index])

    st.subheader('Phòng')
    phong_choices = st.columns(2)
    with phong_choices[0]:
        mmb = st.checkbox('114')
    with phong_choices[1]:
        mmn = st.checkbox('115')
    if mmb: phong.append('114')
    if mmn: phong.append('115')
    phong_check = np.array([1 if df.loc[i, 'PC'] in phong else 0 for i in df.index])

    st.subheader('Lớp chuyên')
    chuyen_choices = st.columns(5)
    with chuyen_choices[0]:
        van = st.checkbox('Văn')
        toan = st.checkbox('Toán')
    with chuyen_choices[1]:
        ly = st.checkbox('Lý')
        hoa = st.checkbox('Hóa')
    with chuyen_choices[2]:
        anh = st.checkbox('Anh')
        tin = st.checkbox('Tin')
    with chuyen_choices[3]:
        sudia = st.checkbox('Sử Địa')
        trungnhat = st.checkbox('Trung Nhật')
    with chuyen_choices[4]:
        thsn = st.checkbox('TH/SN')
        khac = st.checkbox('Khác')
    if van: lopchuyen.append('Chuyên Văn')
    if toan: lopchuyen.append('Chuyên Toán')
    if ly: lopchuyen.append('Chuyên Lý')
    if hoa: lopchuyen.append('Chuyên Hóa')
    if anh: lopchuyen.append('Chuyên Anh')
    if sudia: lopchuyen.append('Sử Địa')
    if trungnhat: lopchuyen.append('Trung Nhật')
    if thsn: lopchuyen.append('Tích Hợp/Song Ngữ')
    if khac: lopchuyen.append('Khác')
    lopchuyen_check = np.array([1 if df.loc[i, 'CLASS-GROUP'] in lopchuyen else 0 for i in df.index])

    display = df[(gender_check == 1) & (grade_check == 1) & (period_check == 1) & (phong_check == 1) & (lopchuyen_check == 1)]

    if st.button('Hiển thị danh sách'):
        st.dataframe(display, use_container_width=True, hide_index=True)

with tab2:
    bdtab1, bdtab2 = st.tabs(["Số lượng học sinh", "Điểm"])
    with bdtab1:
        st.plotly_chart(nam_nu)
        with st.expander("Kết luận"):
            st.write('Nhìn chung học sinh nam quan tâm khóa học hơn học sinh nữ.')
        st.plotly_chart(khoilop)
        with st.expander("Kết luận"):
            st.write('Nhìn chung học sinh càng lớn thì càng ít quan tâm đến khóa học hơn.')
        st.plotly_chart(cg)
        with st.expander("Kết luận"):
            st.write('Học sinh các lớp khối xã hội và ngôn ngữ ít có nhu cầu tham gia khóa học hơn.')
    with bdtab2:
        session_choice = st.radio('Chọn session để hiện biểu đồ điểm theo lớp chuyên', ('S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'GPA'))
        if session_choice == 'S1': st.plotly_chart(box1)
        if session_choice == 'S2': st.plotly_chart(box2)
        if session_choice == 'S3': st.plotly_chart(box3)
        if session_choice == 'S4': st.plotly_chart(box4)
        if session_choice == 'S5': st.plotly_chart(box5)
        if session_choice == 'S6': st.plotly_chart(box6)
        if session_choice == 'S7': st.plotly_chart(box7)
        if session_choice == 'S8': st.plotly_chart(box8)
        if session_choice == 'S9': st.plotly_chart(box9)
        if session_choice == 'S10': st.plotly_chart(box10)
        if session_choice == 'GPA': st.plotly_chart(boxGPA)
        with st.expander("Kết luận"):
            st.write('Kết luận chung từ các biểu đồ trên, Học sinh các lớp Chuyên Tin học tốt nhất ở khóa học này. Ngoài ra có thể thấy các lớp khối tự nhiên học tốt hơn các khối khác.')
        st.plotly_chart(boxBAN)
        st.plotly_chart(hist_cg)
        st.plotly_chart(cont)
        with st.expander("Kết luận"):
            st.write('Nhu cầu tham gia khóa học, MC4AI, giảm đáng kể trong số lượng học sinh đã tham gia khóa Python4AI. Tỷ lệ qua môn của học sinh mỗi loại lớp chuyên là khoảng 60-70 phần trăm.')
with tab3:
    numberofgroups = st.slider('Số nhóm', 2, 5, 2)
    if numberofgroups == 2:
        st.plotly_chart(groups2, use_container_width=True)
        st.write('Nhóm 1: GPA trung bình', X20)
        st.dataframe(df[kmeans2.labels_ == 0][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 2: GPA trung bình', X21)
        st.dataframe(df[kmeans2.labels_ == 1][['NAME', 'CLASS', 'GPA']])
    if numberofgroups == 3:
        st.plotly_chart(groups3, use_container_width=True)
        st.write('Nhóm 1: GPA trung bình', X30)
        st.dataframe(df[kmeans3.labels_ == 0][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 2: GPA trung bình', X31)
        st.dataframe(df[kmeans3.labels_ == 1][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 3: GPA trung bình', X32)
        st.dataframe(df[kmeans3.labels_ == 2][['NAME', 'CLASS', 'GPA']])
    if numberofgroups == 4:
        st.plotly_chart(groups4, use_container_width=True)
        st.write('Nhóm 1: GPA trung bình', X40)
        st.dataframe(df[kmeans4.labels_ == 0][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 2: GPA trung bình', X41)
        st.dataframe(df[kmeans4.labels_ == 1][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 3: GPA trung bình', X42)
        st.dataframe(df[kmeans4.labels_ == 2][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 4: GPA trung bình', X43)
        st.dataframe(df[kmeans4.labels_ == 3][['NAME', 'CLASS', 'GPA']])
    if numberofgroups == 5:
        st.plotly_chart(groups5, use_container_width=True)
        st.write('Nhóm 1: GPA trung bình', X50)
        st.dataframe(df[kmeans5.labels_ == 0][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 2: GPA trung bình', X51)
        st.dataframe(df[kmeans5.labels_ == 1][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 3: GPA trung bình', X52)
        st.dataframe(df[kmeans5.labels_ == 2][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 4: GPA trung bình', X53)
        st.dataframe(df[kmeans5.labels_ == 3][['NAME', 'CLASS', 'GPA']])
        st.write('Nhóm 5: GPA trung bình', X54)
        st.dataframe(df[kmeans5.labels_ == 4][['NAME', 'CLASS', 'GPA']])
with tab4:
    st.pyplot(midavg_plot)
with tab5:
    predict_option = st.selectbox(
    'Chọn hình thức dự đoán',
    ('Trung bình bài tập + giữa kì -> cuối kì', 'Giữa kì + cuối kì -> GPA', 'Giữa kì + trung bình bài tập -> đậu/rớt'))
    if predict_option == 'Trung bình bài tập + giữa kì -> cuối kì':
        avg = st.number_input('Điểm trung bình bài tập', 0.0, 10.0, value=0.0, step=.1)
        giuaki = st.number_input('Điểm giữa kì', 0.0, 10.0, value=0.0, step=1.0)
        test = np.array([[avg, giuaki]])
        pred = model.predict(test).argmax(axis=1)
        if pred == 1: st.write('Điểm cuối kì có khả năng sẽ từ 6 điểm trở lên.')
        if pred == 0: st.write('Điểm cuối kì có khả năng sẽ dưới 6 điểm.')
        st.write('Learning Curve của model dùng để dự đoán:')
        st.pyplot(twoclass)
    if predict_option == 'Giữa kì + cuối kì -> GPA':
        giuaki = st.number_input('Điểm giữa kì', 0, 10, value=0, step=1)
        cuoiki = st.number_input('Điểm cuối kì', 0, 10, value=0, step=1)
        test = np.array([[giuaki, cuoiki]])
        pred = midfin.predict(test)
        st.write('Điểm cuối kì có khả năng sẽ ở khoảng:', pred)
        st.write('Kết quả được dự đoán bằng mô hình linear regression cho các số liệu S6, S10 và GPA:')
        st.plotly_chart(midfin_plane)
    if predict_option == 'Giữa kì + trung bình bài tập -> đậu/rớt':
        avg = st.number_input('Điểm trung bình bài tập', 0.0, 10.0, value=0.0, step=.1)
        giuaki = st.number_input('Điểm giữa kì', 0.0, 10.0, value=0.0, step=1.0)
        test = np.array([[giuaki, avg]])
        pred = midavg.predict(test)
        if pred == 1: st.write('Có khả năng cao sẽ qua môn.')
        if pred == 0: st.write('Có khả năng cao sẽ rớt môn.')
        st.write('Kết quả được dự đoán bằng mô hình logistic regression thông qua các cột số liệu S6, S-AVG và PASS:')
        st.pyplot(midavg_plot)
