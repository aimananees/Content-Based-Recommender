import numpy as np
import pandas as pd

import sklearn
from sklearn.neighbors import NearestNeighbors

cars = pd.read_csv('mtcars.csv')
cars.columns = ['car_names', 'mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear', 'carb']

#Testing-value
t = [15, 300, 160, 3.2] 

#Classification based on the following features(index given)
X = cars.ix[:,(1, 3, 4, 6)].values

#5: 5 nearest neighbours
nbrs = NearestNeighbors(n_neighbors=5).fit(X)

print(nbrs.kneighbors([t]))
