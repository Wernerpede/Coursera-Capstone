
import pandas as pd

import numpy as np

print('Hello Capstone Project Course!')

df = pd.read_csv('Data-Collisions.csv')

columns = df.columns

unknown = df['ROADCOND'].isnull().sum()

correlation = df.corr()
