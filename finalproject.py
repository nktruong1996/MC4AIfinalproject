import numpy as np
import pandas as pd

msg = "Roll a dice"
print(msg)

print(np.random.randint(1,9))

for num in [1,2,3,4,5]:
    print(num)

l = range(5)
a = np.array(l)
print(a)

data = {'name' : ["An",   "Bình", "Châu", "Nam", "Mai"],
        'grade': [   7,      6,       5,     7,     9],
        'class': ['10A1', '10A2', '10A3', '10B', '10C']}
df = pd.DataFrame(data)
print(df)

import streamlit as st

st.title("Hamlet said…")
st.text("""
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. how about now?
""")

df_new = df[df['grade'] >= 7]
print(df_new)
