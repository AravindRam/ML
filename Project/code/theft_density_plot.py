import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns

#to draw the SF map on the background
mapdata = np.loadtxt("sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]

#create the bounding box enclosing the latitude and longitude of SF
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]

train = pd.read_csv('train.csv')

#remove unwanted latitude and longitude
train['Lat'] = train[train.X<-121].X
train['Lon'] = train[train.Y<40].Y
train = train.dropna()
trainL = train[train.Category == 'LARCENY/THEFT']#retrieve only larceny/theft crimes

#Draw the density plot
pl.figure(figsize=(20,20*asp))
ax = sns.kdeplot(trainL.Lat, trainL.Lon, clip=clipsize, aspect=1/asp)
ax.imshow(mapdata, cmap=pl.get_cmap('gray'), extent=lon_lat_box, aspect=asp)
pl.savefig('larceny_density_plot.png') # Save the figure

