
import pandas as pd

# Folium - Visualisation of Geospatial Data

import folium

# Importing Pandas DataFrame and preparing DataFrame

df = pd.read_csv('Data-Collisions.csv')

columns = df.columns

location = df[['SEVERITYCODE','X','Y','SEVERITYDESC','COLLISIONTYPE',
               'JUNCTIONTYPE','WEATHER','ROADCOND','LIGHTCOND']]

# Dropping Nan values from location

location = location.dropna(subset = ['X','Y'])

# Collecting a sample of data from the DataFrame

location = location.sample(600)

# Establishing map location using the mean of the sample.
# Longitude X and Latitude Y

seatle = folium.Map(location = [location.Y.mean(), location.X.mean()],
                    zoom_start = 11.5)

# Arranging the coordintes into separate DataFrames

X = location.X

Y = location.Y

severity_code = location.SEVERITYCODE

location_columns = location.columns

information = location.SEVERITYDESC

# Creating a folium FeatureGroup

feature_group = folium.FeatureGroup('Locations')

for lat, long, code, name in zip(Y,X,severity_code,information):

    if code == 1:

        feature_group.add_child(folium.Marker(
        location = [lat,long], popup = name,
        icon = folium.Icon(color = 'cadetblue', icon = 'info-sign')))
    else:

        feature_group.add_child(folium.Marker(
            location = [lat,long], popup = name,
            icon = folium.Icon(color = 'red', icon = 'info-sign')))

# Saving map as an HTML

seatle = seatle.add_child(feature_group)

seatle = seatle.save('seatle.html')
