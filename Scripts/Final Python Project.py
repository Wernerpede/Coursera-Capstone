# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 17:49:15 2020

@author: guiwe
"""

#%% Libraries Imported

# Code written on Anaconda Spyder

# Importing Python Libraries

import numpy as np
import pandas as pd

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.externals.six import StringIO
import pydotplus
from sklearn import tree

# Geospatial Representation Libraries

import folium

#%% Data Cleaning


# Importing Pandas DataFrame from computer Library

df = pd.read_csv('Data-Collisions.csv')

df_original = df

# Preparing DataFrame for Analysis

columns = df.columns

df = df.drop(columns = ['OBJECTID','INCKEY','COLDETKEY','REPORTNO','STATUS',
                        'INTKEY','EXCEPTRSNDESC','EXCEPTRSNCODE','INCDATE',
                        'SDOTCOLNUM', 'SEGLANEKEY', 'CROSSWALKKEY'])

df['INCDTTM'] = pd.to_datetime(df['INCDTTM'], format = '%m/%d/%Y %I:%M:%S %p', errors = 'coerce')

df['TIME'] = df['INCDTTM'].dt.time

df['HOUR'] = df['INCDTTM'].dt.hour

df['DATE'] = df['INCDTTM'].dt.date

df['YEAR'] = df['INCDTTM'].dt.year

df = df.drop(columns = ['INCDTTM'])

# Replace Categorical (non-numeric) with Numeric

df['UNDERINFL'] = df['UNDERINFL'].replace(to_replace = 'Y', value = 1)
df['UNDERINFL'] = df['UNDERINFL'].replace(to_replace = 'N', value = 0)
df['UNDERINFL'] = df['UNDERINFL'].replace(to_replace = '1', value = 1)
df['UNDERINFL'] = df['UNDERINFL'].replace(to_replace = '0', value = 0)

df['INATTENTIONIND'] = df['INATTENTIONIND'].replace(to_replace = np.nan, value = 0)
df['INATTENTIONIND'] = df['INATTENTIONIND'].replace(to_replace = 'Y', value = 1)

df['PEDROWNOTGRNT'] = df['PEDROWNOTGRNT'].replace(to_replace = np.nan, value = 0)
df['PEDROWNOTGRNT'] = df['PEDROWNOTGRNT'].replace(to_replace = 'Y', value = 1)

df['SPEEDING'] = df['SPEEDING'].replace(to_replace = np.nan, value = 0)
df['SPEEDING'] = df['SPEEDING'].replace(to_replace = 'Y', value = 1)

df['HITPARKEDCAR'] = df['HITPARKEDCAR'].replace(to_replace = 'N', value = 0)
df['HITPARKEDCAR'] = df['HITPARKEDCAR'].replace(to_replace = 'Y', value = 1)

# Encoding the Categorical (non-numerical) columns to Numerical

labelencoder = LabelEncoder()

df = df.dropna() #subset = ['JUNCTIONTYPE','WEATHER','ROADCOND','LIGHTCOND','LOCATION','ADDRTYPE']

df = df.drop(df.index[(df['ROADCOND'] == 'Unknown') | (df['ROADCOND'] == 'Other')].tolist())
df = df.drop(df.index[(df['WEATHER'] == 'Unknown')])
df = df.drop(df.index[(df['LIGHTCOND'] == 'Unknown')])
df = df.drop(df.index[(df['JUNCTIONTYPE'] == 'Unknown')])

df['JUNCTIONTYPE'] = df['JUNCTIONTYPE'].apply(str)
df['COLLISIONTYPE'] = df['COLLISIONTYPE'].apply(str)
df['WEATHER'] = df['WEATHER'].apply(str)
df['ROADCOND'] = df['ROADCOND'].apply(str)
df['LIGHTCOND'] = df['LIGHTCOND'].apply(str)
df['ADDRTYPE'] = df['ADDRTYPE'].apply(str)
df['LOCATION'] = df['LOCATION'].apply(str)

df['JUNCTIONTYPE_CODE'] = labelencoder.fit_transform(df['JUNCTIONTYPE']).astype('int64')
df['COLLISIONTYPE_CODE'] = labelencoder.fit_transform(df['COLLISIONTYPE']).astype('int64')
df['WEATHER_CODE'] = labelencoder.fit_transform(df['WEATHER']).astype('int64')
df['ROADCOND_CODE'] = labelencoder.fit_transform(df['ROADCOND']).astype('int64')
df['LIGHT_CODE'] = labelencoder.fit_transform(df['LIGHTCOND']).astype('int64')
df['ADDRTYPE_CODE'] = labelencoder.fit_transform(df['ADDRTYPE']).astype('int64')
df['LOCATION_CODE'] = labelencoder.fit_transform(df['LOCATION']).astype('int64')

df['ST_COLCODE'] = df['ST_COLCODE'].astype('int64')

# Creating a new Numeric DataFrame for analysis using Machine Learning

df2 = df

df2 = df2.drop(columns = ['LOCATION','SEVERITYDESC','COLLISIONTYPE',
                          'SDOT_COLDESC','WEATHER','ROADCOND',
                          'LIGHTCOND','ST_COLDESC','TIME','DATE',
                          'JUNCTIONTYPE','ADDRTYPE','X','Y'])

# Creating Subsets of DataSet for Analysis

Pedestrian = df[(df['ST_COLCODE'] == 0) | # By location
                (df['ST_COLCODE'] == 1) | # Clear correlation between pedestrian right of way and intersection
                (df['ST_COLCODE'] == 2) | # Clear correlation between vehicle turning and intersection
                (df['ST_COLCODE'] == 3) | # Clear correlation between vehicle going straight and hitting pedestrian,
                (df['ST_COLCODE'] == 4) ] # this may be due to the fact that one vehicle grants right of passage and the other, unaware hits pedestrian.


df6 = Pedestrian.drop(columns = ['LOCATION','SEVERITYDESC','COLLISIONTYPE',
                                        'SDOT_COLDESC','WEATHER','ROADCOND',
                                        'LIGHTCOND','ST_COLDESC','TIME','DATE',
                                        'JUNCTIONTYPE','ADDRTYPE','X','Y'])

Train = df[(df['ST_COLCODE'] == 40) | # Photo of location
           (df['ST_COLCODE'] == 41) |
           (df['ST_COLCODE'] == 42) |
           (df['ST_COLCODE'] == 43) ]

Bicycle = df[(df['ST_COLCODE'] == 44) | # By location
             (df['ST_COLCODE'] == 45) |
             (df['ST_COLCODE'] == 46) |
             (df['ST_COLCODE'] == 5)  ]

df15 = Bicycle.drop(columns = ['LOCATION','SEVERITYDESC','COLLISIONTYPE_CODE',
                               'SDOT_COLDESC','WEATHER','ROADCOND',
                               'LIGHTCOND','ST_COLDESC','TIME','DATE',
                               'JUNCTIONTYPE','ADDRTYPE','X','Y'])

Object = df[(df['ST_COLCODE'] == 50) |
            (df['ST_COLCODE'] == 51) |
            (df['ST_COLCODE'] == 52) ]

Animal = df[(df['ST_COLCODE'] == 47) |
            (df['ST_COLCODE'] == 48) |
            (df['ST_COLCODE'] == 49) ]

Machinery = df[(df['ST_COLCODE'] == 60) | # By location
               (df['ST_COLCODE'] == 61) |
               (df['ST_COLCODE'] == 62) |
               (df['ST_COLCODE'] == 63) |
               (df['ST_COLCODE'] == 64) |
               (df['ST_COLCODE'] == 65) |
               (df['ST_COLCODE'] == 66) |
               (df['ST_COLCODE'] == 67) ]

Hazard = df[(df['ST_COLCODE'] == 54) | # low fire
            (df['ST_COLCODE'] == 55) ]

Drugs = df[(df['UNDERINFL'] == 1)]

df_Drugs = Pedestrian.drop(columns = ['LOCATION','SEVERITYDESC','COLLISIONTYPE',
                                        'SDOT_COLDESC','WEATHER','ROADCOND',
                                        'LIGHTCOND','ST_COLDESC','TIME','DATE',
                                        'JUNCTIONTYPE','ADDRTYPE','X','Y'])

Parking = df[(df['ST_COLCODE'] == 10) | # By locaiton & type
             (df['ST_COLCODE'] == 19) | # Clear correlation between midblock collisions
             (df['ST_COLCODE'] == 20) |
             (df['ST_COLCODE'] == 21) |
             (df['ST_COLCODE'] == 22) |
             (df['ST_COLCODE'] == 32)]

Opposite_Direction = df[(df['ST_COLCODE'] == 24) | # Under influence of alcohol or drugs?
                        (df['ST_COLCODE'] == 25) | # Left Turn and Intersection, Clear Correlation
                        (df['ST_COLCODE'] == 26) |
                        (df['ST_COLCODE'] == 27) |
                        (df['ST_COLCODE'] == 28) |
                        (df['ST_COLCODE'] == 29) |
                        (df['ST_COLCODE'] == 30) ]

Same_Direction = df[(df['ST_COLCODE'] == 71) | # Distraction?
                    (df['ST_COLCODE'] == 72) |
                    (df['ST_COLCODE'] == 73) |
                    (df['ST_COLCODE'] == 74) |
                    (df['ST_COLCODE'] == 81) |
                    (df['ST_COLCODE'] == 82) |
                    (df['ST_COLCODE'] == 83) |
                    (df['ST_COLCODE'] == 84) ]

Breaking_Parts = df[df['ST_COLCODE'] == 56]

Innatention = df[df['INATTENTIONIND'] == 1] # Telephone?

unknown = df['ROADCOND'].isnull().sum()

correlation = df.corr()

#%% Geospatial Plot Folium

# Retrieving a sample of 600 points from the data, to plot on map
location = df[['LOCATION','X','Y','SEVERITYCODE']].sample(600)

seatle = folium.Map(location = [location.Y.mean(), location.X.mean()],
                    zoom_start = 11.5)

X_cord = location.X

Y_cord = location.Y

severity_code = location.SEVERITYCODE

street_name = location.LOCATION

# Creating a folium FeatureGroup

feature_group = folium.FeatureGroup('Locations')

for lat, long, loc, code in zip(Y_cord, X_cord, street_name, severity_code):

    if code == 1:

        feature_group.add_child(folium.Marker(
        location = [lat,long], popup = loc,
        icon = folium.Icon(color = 'green', icon = 'info-sign')))
    else:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = loc,
            icon = folium.Icon(color = 'red', icon = 'info-sign')))

# Saving map as an HTML

seatle = seatle.add_child(feature_group)

seatle = seatle.save('seatle_folium.html')

#%% Speeding Correlations and Plots

# Conditional probability P(Injury|Speeding)

Injury_Speeding = df[(df['SEVERITYCODE.1'] == 2) & (df['SPEEDING'] == 1)] # P(Injury n Speeding)

prob_1 = Injury_Speeding.shape[0]/df['SPEEDING'].sum() # P(Speeding)

# Conditional probability P(Injury|Not Speeding)

Injury_not_Speeding = df[(df['SEVERITYCODE.1'] == 2) & (df['SPEEDING'] == 0)] # P(Injury n Not Speeding)

prob_2 = Injury_not_Speeding.shape[0]/(df.shape[0] - df['SPEEDING'].sum()) # P(Not Speeding)

# Injury Prevalance by Location

injury_location = Injury_Speeding.groupby(['LOCATION'])['SPEEDING'].sum().sort_values(ascending = False).head(50) # Determining injury prevalance by location.

# Plot of Speeding Categorised according to Road Condition and Light Condition

bar_plot = df.groupby(['ROADCOND','LIGHTCOND']).sum()['SPEEDING'].unstack().fillna(0).plot(kind = 'bar', fontsize = 13, width = 1.0)
plt.title('Speeding vehicle correlation with Road Condition and Lighting Condition')
plt.xlabel('Road Condition')
plt.ylabel('Number of Collisions (2004-2020)')
plt.legend(loc = 10, fontsize = 'small', title = 'Light Condition')
plt.show()

# Plot of speeding prevalance against loaction (horizontal bar plot (barh))

df.groupby(['LOCATION'])['SPEEDING'].sum().sort_values(ascending = False).head(12).plot(kind = 'barh')
plt.title('Locations with highest accounts of Speeding')
plt.xlabel('Number of Incidents')
plt.ylabel('Location Address')
plt.show()

# Heatmap of Speeding against Road Condition, Co-Ocurrance Matrix

df3 = df

df3['SPEEDING'] = df3['SPEEDING'].replace(0, 'No', regex = True)
df3['SPEEDING'] = df3['SPEEDING'].replace(1, 'Yes', regex = True)

corr_matrix_Speeding = pd.crosstab(index = df3.ROADCOND, columns = df3.SPEEDING)

sns.heatmap(corr_matrix_Speeding, annot = True, cmap = 'Oranges', fmt = 'd')
plt.title('Heatmap of Speeding against Road Condition (2004-2020)')
plt.ylabel('Road Condition')
plt.xlabel('Speeding')
plt.show()

#%% Accidents per time of Day Histogram

# Histogram Time of Day Plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
fig = plt.hist(df3['HOUR'], bins = 50)
# Plot Title and Labels
plt.xlabel('Time of Day')
plt.ylabel('Accidents')
plt.title('Accidents per Time of Day')
plt.xlim([0,23])
# Major ticks every 6 minor ticks every 1
major_ticks = np.arange(0, 25, 6)
minor_ticks = np.arange(0, 25, 1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
# And a corresponding grid
ax.grid(which='both')
# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
plt.show()

#%% Drugs and Alcohol Machine Learning


df_Drugs['UNDERINFL'] = df_Drugs['UNDERINFL'].replace(1, 'yes', regex = True)
df_Drugs['UNDERINFL'] = df_Drugs['UNDERINFL'].replace(0, 'no', regex = True)

y = (df_Drugs['UNDERINFL'] == 'yes')  # Converting to Binary

feature_names = [i for i in df_Drugs.columns if df_Drugs[i].dtype in [np.int64]]

X = df_Drugs[feature_names]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

clf = RandomForestClassifier(n_estimators = 100, random_state = 0).fit(x_train, y_train)

perm = PermutationImportance(clf, random_state = 1).fit(x_test, y_test)

html_obj = eli5.show_weights(perm, feature_names = x_train.columns.tolist())

with open("drugs_and_alcohol_machine_learning.html","wb") as f:
    f.write(html_obj.data.encode("UTF-8"))

# max_depth can be established max_depth = () for the decision tree.

# Calculating the RandomForestTree Error
# Requires importing of new sklearn library
    # from sklearn.metrics import mean_absolute_error
# pred = clf.predict(x_test)
# error = mean_absolute_error(pred, y_test)

#%% Co-Ocurrange Matrix  (Heat Map) of Speeding against Alcohol

df3['UNDERINFL'] = df3['UNDERINFL'].replace(0, 'No', regex = True)
df3['UNDERINFL'] = df3['UNDERINFL'].replace(1, 'Yes', regex = True)

corr_matrix_Speeding = pd.crosstab(index = df3.UNDERINFL, columns = df3.SPEEDING)

# Plot 1

sns.heatmap(corr_matrix_Speeding, annot = True, cmap = 'YlGnBu_r', fmt = 'd')
plt.title('Heatmap of Speeding and Alcohol/Drug Consumption (2004-2020)')
plt.ylabel('Under Influence of Alcohol or Drugs')
plt.xlabel('Speeding')
plt.show()

# Plot 2

corr_matrix_Speeding = pd.crosstab(index = df3.UNDERINFL, columns = df3.LIGHTCOND)

sns.heatmap(corr_matrix_Speeding, annot = True, cmap = 'Blues', fmt = 'd')
plt.title('Heatmap of Alcohol/Drug Consumption with Light Condition (2004-2020)')
plt.ylabel('Under Influence of Alcohol or Drugs')
plt.xlabel('Light Condition')
plt.show()

# Plot 3

corr_matrix_Speeding = pd.crosstab(index = df3.UNDERINFL, columns = df3.JUNCTIONTYPE)

sns.heatmap(corr_matrix_Speeding, annot = True, cmap = 'Blues', fmt = 'd')
plt.title('Heatmap of Alcohol/Drug Consumption with Junction Type (2004-2020)')
plt.ylabel('Under Influence of Alcohol or Drugs')
plt.xlabel('Junction Type')
plt.show()



#%% Accidents per time of Day Alcohol and Drugs

# Recommended Police Blitz times

# Histogram Time of Day Plot
fig = plt.figure()

df20 = df3[df3['UNDERINFL'] == 'Yes']
df20['HOUR'] = df20['HOUR'].replace(0, 24, regex = True)

ax = fig.add_subplot(1, 1, 1)
fig = plt.hist(df20['HOUR'], bins = 50)

# Plot Title and Labels
plt.xlabel('Time of Day')
plt.ylabel('Drugs/Alcohol Accidents')
plt.title('Drugs/Alcohol Accidents per Time of Day')
plt.xlim([1,23])
# Major ticks every 6 minor ticks every 1
major_ticks = np.arange(0, 25, 6)
minor_ticks = np.arange(0, 25, 1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks, minor=True)
# And a corresponding grid
ax.grid(which='both')
# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
plt.show()

#%% Alcohol and Drugs Geospatial Data

alcohol_drugs = Drugs.groupby(['LOCATION','X','Y'])['VEHCOUNT'].count().sort_values(ascending=False).to_frame()

alcohol_drugs = alcohol_drugs.reset_index()

alcohol_drugs = alcohol_drugs[alcohol_drugs['VEHCOUNT'] > 5]

alcohol_drugs['X'] = alcohol_drugs['X'].astype('float')
alcohol_drugs['Y'] = alcohol_drugs['Y'].astype('float')

X_cord = alcohol_drugs.X

Y_cord = alcohol_drugs.Y

vehicle_count = alcohol_drugs.VEHCOUNT

people = folium.Map(location = [location.Y.mean(), location.X.mean()],
                    zoom_start = 11.5)

# Creating a folium FeatureGroup

feature_group = folium.FeatureGroup('people')

for lat, long, acc in zip(Y_cord, X_cord, vehicle_count):

    if 5 < acc < 8 :

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'green', icon = 'info-sign')))

    elif 8 < acc < 11:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'orange', icon = 'info-sign')))

    else:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'red', icon = 'info-sign')))

# Saving map as an HTML

people = people.add_child(feature_group)

people = people.save('drugs_folium.html')

#%% Decision Tree Alcohol and Drugs

df5 = df2

df5['UNDERINFL'] = df5['UNDERINFL'].replace(1, 'yes', regex = True)
df5['UNDERINFL'] = df5['UNDERINFL'].replace(0, 'no', regex = True)

# Making the address type into binary data
df5 = df5.drop(df.index[(df['JUNCTIONTYPE'] == 2)])

# Decision Tree Run Solely on Binary Data
X = df5[['LIGHT_CODE','COLLISIONTYPE_CODE','JUNCTIONTYPE_CODE',
         'WEATHER_CODE','ROADCOND_CODE','ADDRTYPE_CODE']].values


# The comment box below indicates another decision tree possibility, for binary data
# ['INATTENTIONIND','PEDROWNOTGRNT','SPEEDING','HITPARKEDCAR']

y = df5['UNDERINFL']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state = 3)

# Max depth established at max_depth = 4

drugTree = DecisionTreeClassifier(criterion = 'entropy', max_depth= 4).fit(x_train, y_train)

# Testing the decision Tree Model with the x_test data
predTree = drugTree.predict(x_test)

# Both of the below metrics should yield the same score.
# Validating the accuracy of the model by comparing the above prediction to the y_test (actual values)
accuracy = metrics.accuracy_score(y_test, predTree)
# Scoring the accuracy of the model.
score = drugTree.score(x_test, y_test)

# Decision Tree

dot_data = StringIO()
filename = 'drugtree.png'
featureNames = ['LIGHT_CODE','COLLISIONTYPE_CODE','JUNCTIONTYPE_CODE',
                'WEATHER_CODE','ROADCOND_CODE','ADDRTYPE_CODE']

targetNames = y.unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames,
                         out_file=dot_data, class_names= np.unique(y_train),
                         filled=True, special_characters=True, rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

#%% Pedestrian Machine Learning

# Pedestrian Right of Passage

df6['PEDROWNOTGRNT'] = df6['PEDROWNOTGRNT'].replace(1, 'yes', regex = True)
df6['PEDROWNOTGRNT'] = df6['PEDROWNOTGRNT'].replace(0, 'no', regex = True)

y = (df6['PEDROWNOTGRNT'] == 'yes')  # Converting to Binary

feature_names = [i for i in df6.columns if df6[i].dtype in [np.int64]]

X = df6[feature_names]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

clf = RandomForestClassifier(n_estimators = 100, random_state = 0).fit(x_train, y_train)

perm = PermutationImportance(clf, random_state = 1).fit(x_test, y_test)

html_obj = eli5.show_weights(perm, feature_names = x_train.columns.tolist())

with open("pedestrian_machine_learning.html","wb") as f:
    f.write(html_obj.data.encode("UTF-8"))

#%% Pedestrian Co-Ocurrance Matrix

# Pedestrian Right of Passage and Collision Type

Pedestrian['PEDROWNOTGRNT'] = Pedestrian['PEDROWNOTGRNT'].replace(1, 'yes', regex = True)
Pedestrian['PEDROWNOTGRNT'] = Pedestrian['PEDROWNOTGRNT'].replace(0, 'no', regex = True)

corr_matrix = pd.crosstab(index = Pedestrian.ST_COLDESC, columns = Pedestrian.PEDROWNOTGRNT)

sns.heatmap(corr_matrix, cmap = 'Blues', annot = True, fmt = 'd')
plt.title('Heatmap of Pedestrian Right of Passage and Collision Type (2004-2020)')
plt.ylabel('Collision Type')
plt.xlabel('Pedestrian Right of Passage Granted')
plt.show()

# Pedestrian Right of Passage and Innatention

Pedestrian['INATTENTIONIND'] = Pedestrian['INATTENTIONIND'].replace(1, 'yes', regex = True)
Pedestrian['INATTENTIONIND'] = Pedestrian['INATTENTIONIND'].replace(0, 'no', regex = True)

corr_matrix = pd.crosstab(index = Pedestrian.INATTENTIONIND, columns = Pedestrian.PEDROWNOTGRNT)

sns.heatmap(corr_matrix, cmap = 'Blues', annot = True, fmt = 'd')
plt.title('Heatmap of Pedestrian Right of Passage and Inattentiveness Type (2004-2020)')
plt.ylabel('Inattentiveness')
plt.xlabel('Pedestrian Right of Passage Granted')
plt.show()


#%% Pedestrians Geospatial Plot

pedestrians = df.groupby(['LOCATION','X','Y'])['PEDCOUNT'].count().sort_values(ascending=False).to_frame()

pedestrians = pedestrians.reset_index()

pedestrians = pedestrians[pedestrians['PEDCOUNT'] > 60]

pedestrians['X'] = pedestrians['X'].astype('float')
pedestrians['Y'] = pedestrians['Y'].astype('float')

X_cord = pedestrians.X

Y_cord = pedestrians.Y

pedstrian_count = pedestrians.PEDCOUNT

people = folium.Map(location = [location.Y.mean(), location.X.mean()],
                    zoom_start = 11.5)

# Creating a folium FeatureGroup

feature_group = folium.FeatureGroup('people')

for lat, long, acc in zip(Y_cord, X_cord, pedstrian_count):

    if 60 < acc < 100 :

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'lightgreen', icon = 'info-sign')))

    elif 99 < acc < 150:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'darkgreen', icon = 'info-sign')))

    elif 149 < acc < 200:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'orange', icon = 'info-sign')))

    else:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'red', icon = 'info-sign')))

# Saving map as an HTML

people = people.add_child(feature_group)

people = people.save('pedestrian_folium.html')

parking_spaces = df[df['LOCATION'] == 'AURORA AVE N BETWEEN N 130TH ST AND N 135TH ST']

parking_spaces = parking_spaces.groupby(['ST_COLDESC'])['PEDCOUNT'].count().sort_values(ascending=False).to_frame()

#should be analysed?
accident = df[df['ST_COLDESC'] == 'From same direction - both going straight - one stopped - rear-end']
abba = accident.PEDROWNOTGRNT.sum()
alla = accident.PEDCOUNT.sum()


# Innatention
innatentive = df.groupby(['LOCATION','X','Y'])['INATTENTIONIND'].sum().sort_values(ascending=False).to_frame()

#%% Bicycle Geospatial Plot

bicycle = df.groupby(['LOCATION','X','Y'])['PEDCYLCOUNT'].count().sort_values(ascending=False).to_frame()

bicycle = bicycle.reset_index()

bicycle['X'] = bicycle['X'].astype('float')
bicycle['Y'] = bicycle['Y'].astype('float')

bicycle = bicycle[bicycle['PEDCYLCOUNT'] > 60]

X_cord = bicycle.X

Y_cord = bicycle.Y

bicycle_count = bicycle.PEDCYLCOUNT

people = folium.Map(location = [Y_cord.mean(), X_cord.mean()],
                    zoom_start = 11.5)

# Creating a folium FeatureGroup

feature_group = folium.FeatureGroup('people')

for lat, long, acc in zip(Y_cord, X_cord, bicycle_count):

    if 60 < acc < 100 :

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'lightgreen', icon = 'info-sign')))

    elif 99 < acc < 150:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'darkgreen', icon = 'info-sign')))

    elif 149 < acc < 200:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'orange', icon = 'info-sign')))

    else:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = acc,
            icon = folium.Icon(color = 'red', icon = 'info-sign')))

# Saving map as an HTML

people = people.add_child(feature_group)

people = people.save('Bicycles.html')

# Additional Analysis on Parking Spaces

# parking_spaces = df[df['LOCATION'] == 'AURORA AVE N BETWEEN N 130TH ST AND N 135TH ST']

# parking_spaces = parking_spaces.groupby(['ST_COLDESC'])['PEDCOUNT'].count().sort_values(ascending=False).to_frame()

# #should be analysed?
# accident = df[df['ST_COLDESC'] == 'From same direction - both going straight - one stopped - rear-end']
# abba = accident.PEDROWNOTGRNT.sum()
# alla = accident.PEDCOUNT.sum()


# # Innatention
# innatentive = df.groupby(['LOCATION','X','Y'])['INATTENTIONIND'].sum().sort_values(ascending=False).to_frame()


#%% Heat Maps with Additional Corellations

# Co-Occurance Matrix for non-numeric variables

corr_matrix = pd.crosstab(index = df.JUNCTIONTYPE, columns = df.SEVERITYDESC)

sns.heatmap(corr_matrix, cmap = 'Blues', annot = True, fmt = 'd')
plt.title('Heatmap of Severity against Junction Type (2004-2020)')
plt.ylabel('Junction Type')
plt.xlabel('Severity')
plt.show()

mid_block = df[(df['JUNCTIONTYPE'] == 'Mid-Block (not related to intersection)')
               & (df['SEVERITYDESC'] == 'Property Damage Only Collision')
               & (df['COLLISIONTYPE'] == 'Parked Car')]

mid_parked = pd.crosstab(index = mid_block['LOCATION'], columns = df['HITPARKEDCAR']).sort_values(by = 1,ascending = False)
mid_parked_2 = pd.crosstab(index = mid_block['SDOT_COLDESC'], columns = df['ST_COLDESC'])

# Was the person on the telephone?

inatention_pedest = df[df['LOCATION'] == 'N NORTHGATE WAY BETWEEN MERIDIAN AVE N AND CORLISS AVE N']

corr_matrix = pd.crosstab(index = df.PEDCOUNT, columns = df.INATTENTIONIND)

sns.heatmap(corr_matrix, cmap = 'Blues', annot = True, fmt = 'd')
plt.title('Heatmap of Pedestiran and Inattention (2004-2020)')
plt.ylabel('Pedestrian Count')
plt.xlabel('Inattention')
plt.show()


#%% Correlations Related to Tunnels

# Co-Occurance Matrix for non-numeric variables

corr_matrix_2 = pd.crosstab(index = df.JUNCTIONTYPE, columns = df.SDOT_COLDESC)

sns.heatmap(corr_matrix_2, cmap = sns.diverging_palette(220, 20, n=7))

plt.show()

address_detailed = df.groupby(['LOCATION'])['X','Y'].mean()

vehicles = df.groupby(['LOCATION'])['VEHCOUNT'].count().sort_values(ascending=False).head(20)

tunnel_1 = df_original[df_original.LOCATION == 'BATTERY ST TUNNEL NB BETWEEN ALASKAN WY VI NB AND AURORA AVE N']

tunnel_2 = df_original[df_original.LOCATION == 'BATTERY ST TUNNEL SB BETWEEN AURORA AVE N AND ALASKAN WY VI SB']

tunnel = tunnel_1.append(tunnel_2)

tunnel_type = tunnel.groupby(['ROADCOND'])['VEHCOUNT'].count()

ax = plt.subplot(111)

tunnel_type.plot.barh()

plt.show()

corr_matrix_3 = pd.crosstab(index = tunnel.SDOT_COLDESC, columns = tunnel.ROADCOND)

tunnel_dry = tunnel[tunnel['ROADCOND'] == 'Dry'].groupby(['SPEEDING'])['VEHCOUNT'].count()

df5 = df

#%% More Correlations, Plots, Testing, and Experimentation


crash_times = df5.groupby(['HOUR']).VEHCOUNT.sum()

crash_days = df5.groupby(['DATE']).VEHCOUNT.sum().sort_values(ascending = False)

current = df5.groupby(['DATE']).VEHCOUNT.sum().tail(200)

corr_matrix_6 = pd.crosstab(index = df5.HOUR, columns = df5.LIGHTCOND)
sns.heatmap(corr_matrix_6, annot = True, cmap = sns.diverging_palette(220, 20, n=7), fmt = 'd')
plt.show()

location_streetlight = pd.crosstab(index = df5.LOCATION, columns = df5.LIGHTCOND)
# It could be argued that there are less streats which lack street lighting, or it could be the case
# that drivers are usually more cautious when street lighting is off, driving at a lower speed.

pedestrian_times = df5.groupby(['HOUR']).PEDCOUNT.sum()

sns.catplot(x = 'HOUR', y = 'VEHCOUNT', data = df5, kind = 'violin')

plt.show()

crash_year = df5.groupby(['YEAR']).VEHCOUNT.sum()

crash_year = crash_year.reset_index()

crash_year.plot.scatter(x = 'YEAR', y = 'VEHCOUNT')

plt.show()

mean = df5.groupby(['YEAR']).VEHCOUNT.sum().mean()


# Annual Plot, does not add much to the investigation

df6 = df5[df5.YEAR != 2020]
crash_types = df6.groupby(['YEAR'])['VEHCOUNT','PEDCOUNT','PEDCYLCOUNT','PERSONCOUNT'].sum()
crash_types = crash_types.reset_index()
fig = plt.figure()
ax = fig.add_subplot(111)
w = 0.2
ax.bar(crash_types.YEAR - 0.4, crash_types.VEHCOUNT, width = w, align = 'center')
ax.bar(crash_types.YEAR - w, crash_types.PEDCOUNT, width = w, align = 'center')
ax.bar(crash_types.YEAR, crash_types.PEDCYLCOUNT, width = w, align = 'center')
ax.bar(crash_types.YEAR + w, crash_types.PERSONCOUNT, width = w, align = 'center')
ax.legend()
ax.autoscale()
plt.legend()
plt.show()




df2 = df.groupby(['WEATHER'])['VEHCOUNT'].max()

ax = plt.gca()

ax = df2.plot.barh()

for i, v in enumerate(df2):

   ax.text(v + 0, i - 0.1, str(v), color = 'blue', ha = 'left')

plt.xlabel('Total Number of Vehicles involved in Collision')

plt.show()
