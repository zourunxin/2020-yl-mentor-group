from calendar import c
from sklearn.cluster import KMeans
import pandas as pd


# load data from pkg_info.csv
data = pd.read_csv('../resources/pkg_info.csv')
data = data.set_index('pkg_name')

# remove 'desc_keyword' and 'label' column in data
train_data = data.drop(labels = ['desc_keyword', 'label'], axis = 1);

# build a k-means cluster model
model = KMeans(n_clusters = 11)

# fit the model
model.fit(train_data)

prediction = model.predict(train_data)
data.loc[:, 'prediction'] = prediction

# sort data by label
data = data.sort_values(by='label')

# write data to file
data.to_csv('./k-means_results.csv')
print(data)










    