# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:49:15 2020

@author: guiwe
"""

#%% Libraries

# Code written on Anaconda Spyder

# Importing Python Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#%% Data Cleaning


# Importing Pandas DataFrame from computer Library

df = pd.read_csv('Data-Collisions.csv')

# Dropping data which will not be used

df = df.drop(columns = ['OBJECTID','INCKEY','COLDETKEY','REPORTING','STATUS',
                        'INTKEY','EXCEPTRSNDESC','INCDATE','SDOTCLNUM','SEGALANEKEY'])

# df5['INCDTTM'] = pd.to_datetime(df5['INCDTTM'], format = '%m/%d/%Y %I:%M:%S %p', errors = 'coerce')

# df5['time'] = df5['INCDTTM'].dt.time

# df5['hour'] = df5['INCDTTM'].dt.hour

# df5['date'] = df5['INCDTTM'].dt.date

# df5['year'] = df5['INCDTTM'].dt.year

columns = df.columns

unknown = df['ROADCOND'].isnull().sum()

correlation = df.corr()

#%% Speeding

# Preparing Speeding Data

df2 = df

df2['SPEEDING'] = df2['SPEEDING'].fillna(0)
df2['SPEEDING'] = df2['SPEEDING'].replace('Y', 1, regex = True)

# Conditional probability P(Injury|Speeding)

Injury_Speeding = df2[(df2['SEVERITYCODE.1'] == 2) & (df2['SPEEDING'] == 1)] # P(Injury n Speeding)

prob_1 = Injury_Speeding.shape[0]/df2['SPEEDING'].sum() # P(Speeding)

# Conditional probability P(Injury|Not Speeding)

Injury_not_Speeding = df2[(df2['SEVERITYCODE.1'] == 2) & (df2['SPEEDING'] == 0)] # P(Injury n Not Speeding)

prob_2 = Injury_not_Speeding.shape[0]/(df2.shape[0] - df2['SPEEDING'].sum()) # P(Not Speeding)

# Injury Prevalance by Location

injury_location = Injury_Speeding.groupby(['LOCATION'])['SPEEDING'].sum().sort_values(ascending = False).head(50)

# Plot of Speeding Categorised according to Road Condition and Light Condition

# Droping 'Unknown' and 'Other' Road Condition category
df2 = df2.drop(df2.index[(df2['ROADCOND'] == 'Unknown') | (df2['ROADCOND'] == 'Other')].tolist())

hello = df2.groupby(['ROADCOND','LIGHTCOND']).sum()['SPEEDING'].unstack().fillna(0).plot(kind = 'bar', fontsize = 13, width = 1.0)

plt.xlabel('Road Condition')
plt.ylabel('Number of Collisions (2004-2020)')
plt.legend(loc = 10, fontsize = 'x-small', title = 'Light Condition')
plt.show()

# Plot of speeding prevalance against loaction

df2.groupby(['LOCATION'])['SPEEDING'].sum().sort_values(ascending = False).head(12).plot(kind = 'bar')
plt.show()

# Heatmap of Speeding against Road Condition

df2['SPEEDING'] = df2['SPEEDING'].replace(0, 'No', regex = True)
df2['SPEEDING'] = df2['SPEEDING'].replace(1, 'Yes', regex = True)


corr_matrix_Speeding = pd.crosstab(index = df2.ROADCOND, columns = df2.SPEEDING)

sns.heatmap(corr_matrix_Speeding, annot = True, cmap = 'Oranges', fmt = 'd')
plt.title('Heatmap of Speeding against Road Condition (2004-2020)')
plt.ylabel('Road Condition')
plt.xlabel('Speeding')
plt.show()

#%%

# Co-Occurance Matrix for non-numeric variables

df = df.drop(df.index[(df['JUNCTIONTYPE'] == 'Unknown')].tolist())

corr_matrix = pd.crosstab(index = df.JUNCTIONTYPE, columns = df.SEVERITYDESC)

sns.heatmap(corr_matrix, cmap = 'Blues', annot = True, fmt = 'd')
plt.title('Heatmap of Severity against Junction Type (2004-2020)')
plt.ylabel('Junction Type')
plt.xlabel('Severity')

mid_block = df[(df['JUNCTIONTYPE'] == 'Mid-Block (not related to intersection)')
               & (df['SEVERITYDESC'] == 'Property Damage Only Collision')
               & (df['COLLISIONTYPE'] == 'Parked Car')]

mid_parked = pd.crosstab(index = mid_block['LOCATION'], columns = df['HITPARKEDCAR']).sort_values(by = 'Y',ascending = False)
mid_parked_2 = pd.crosstab(index = mid_block['SDOT_COLDESC'], columns = df['ST_COLDESC'])


# Was the person on the telephone?

#%%
# ===================================================

# Co-Occurance Matrix for non-numeric variables

corr_matrix_2 = pd.crosstab(index = df.JUNCTIONTYPE, columns = df.SDOT_COLDESC)

sns.heatmap(corr_matrix_2, cmap = sns.diverging_palette(220, 20, n=7))

plt.show()

corr_matrix_2 = pd.crosstab(index = df.LOCATION, columns = df.SEVERITYDESC)

sns.heatmap(corr_matrix_2, cmap = sns.diverging_palette(220, 20, n=7))

plt.show()

ac = df[['LOCATION','X','Y']].dropna()

address_detailed = ac.groupby(['LOCATION'])['X','Y'].mean()

vehicles = df.groupby(['LOCATION'])['VEHCOUNT'].count().sort_values(ascending=False).head(20)

tunnel_1 = df[df.LOCATION == 'BATTERY ST TUNNEL NB BETWEEN ALASKAN WY VI NB AND AURORA AVE N']

tunnel_2 = df[df.LOCATION == 'BATTERY ST TUNNEL SB BETWEEN AURORA AVE N AND ALASKAN WY VI SB']

tunnel = tunnel_1.append(tunnel_2)

tunnel_type = tunnel.groupby(['ROADCOND'])['VEHCOUNT'].count()

ax = plt.subplot(111)

tunnel_type.plot.barh()

plt.show()

corr_matrix_3 = pd.crosstab(index = tunnel.SDOT_COLDESC, columns = tunnel.ROADCOND)

tunnel_dry = tunnel[tunnel['ROADCOND'] == 'Dry'].groupby(['SPEEDING'])['VEHCOUNT'].count()

# ===================================================

# Boxplot
# Catplot

# Based on the location of the element

#df['new_date'] = [d.date() for d in df['INCDTTM']]
#df['new_time'] = [d.time() for d in df['INCDTTM']]


#df['time'] = df['INCDTTM'].map(lambda ts: ts.strftime('%m/%-d/%Y %-H:%M:%S %%'))


#time = df[df['INCDTTM'] == '%m/%-d/%Y %-H:%M:%S']

length = len(df['INCDTTM'][0])

df5 = df

df5['INCDTTM'] = pd.to_datetime(df5['INCDTTM'], format = '%m/%d/%Y %I:%M:%S %p', errors = 'coerce')

df5['time'] = df5['INCDTTM'].dt.time

df5['hour'] = df5['INCDTTM'].dt.hour

df5['date'] = df5['INCDTTM'].dt.date

df5['year'] = df5['INCDTTM'].dt.year

df5['time'] = df5['time'].dropna()

#for d in df['INCDTTM']:

#    dt = datetime.strptime(d, '%m/%d/%Y %H:%M:%S %p')

# df['date'], df['time'] = df['INCDTTM'].split()

crash_times = df5.groupby(['hour']).VEHCOUNT.sum()

crash_days = df5.groupby(['date']).VEHCOUNT.sum().sort_values(ascending = False)

current = df5.groupby(['date']).VEHCOUNT.sum().tail(200)

#
df6 = df5
df6 = df6.drop(df6[(df6.LIGHTCOND == 'Other') | (df6.LIGHTCOND == 'Unknown')].index, inplace = True)
corr_matrix_6 = pd.crosstab(index = df5.hour, columns = df5.LIGHTCOND)
sns.heatmap(corr_matrix_6, annot = True, cmap = sns.diverging_palette(220, 20, n=7), fmt = 'd')
plt.show()
location_streetlight = pd.crosstab(index = df5.LOCATION, columns = df5.LIGHTCOND)
# It could be argued that there are less streats which lack street lighting, or it could be the case
# that drivers are usually more cautious when street lighting is off, driving at a lower speed.
#

pedestrian_times = df5.groupby(['hour']).PEDCOUNT.sum()

sns.catplot(x = 'hour', y = 'VEHCOUNT', data = df5, kind = 'violin')

plt.show()

crash_year = df5.groupby(['year']).VEHCOUNT.sum()

crash_year = crash_year.reset_index()

crash_year.plot.scatter(x = 'year', y = 'VEHCOUNT')

plt.show()

mean = df5.groupby(['year']).VEHCOUNT.sum().mean()

df6 = df5[df5.year != 2020]

crash_types = df6.groupby(['year'])['VEHCOUNT','PEDCOUNT','PEDCYLCOUNT','PERSONCOUNT'].sum()



# ===================================================

# Scatter plot of Time of day against vehicles involved

# ===================================================

# Scatter Plot

df2 = df.groupby(['WEATHER'])['VEHCOUNT'].max()

ax = plt.gca()

ax = df2.plot.barh()

for i, v in enumerate(df2):

   ax.text(v + 0, i - 0.1, str(v), color = 'blue', ha = 'left')

plt.xlabel('Total Number of Vehicles involved in Collision')

plt.show()
