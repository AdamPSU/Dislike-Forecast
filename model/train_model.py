import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

dislikes = pd.read_csv('../data/raw/youtube_dislike_dataset.csv')
dislikes_sub = dislikes[['view_count', 'comment_count', 'likes', 'dislikes']]

# Cross validation
y = dislikes_sub['dislikes']
X = dislikes_sub.drop('dislikes', axis=1)

lr = LinearRegression()
results = cross_val_score(lr, X, y, scoring='neg_root_mean_squared_error', cv=10)
rmse = -results.mean() # RMSE of 20449
