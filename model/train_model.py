import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from itertools import combinations

dislikes = pd.read_json('../data/raw/train.json', lines=True)

# Keep only entries with more than 2,000 subscribers
dislikes = dislikes[dislikes['uploader_sub_count'] > 2_000]

# Select only relevant features
dislikes = dislikes[['upload_date', 'uploader_sub_count', 'view_count',
                     'like_count', 'dislike_count', 'has_subtitles',
                     'is_ads_enabled', 'is_comments_enabled']]

# Encoding
cat = ['has_subtitles', 'is_ads_enabled', 'is_comments_enabled']
dislikes = pd.get_dummies(dislikes, columns=cat, drop_first=True)

# Feature Engineering

# Create a new year variable in favor of the upload date
dislikes['upload_date'] = pd.to_datetime(dislikes['upload_date'], errors='coerce')
dislikes['age'] = (pd.Timestamp.today() - dislikes["upload_date"]).dt.days

dislikes = dislikes.drop('upload_date', axis=1)
dislikes = dislikes.dropna()

# Make interaction terms to better capture the relationship between sub and dislike count
dislikes['uploader_sub_count^2'] = dislikes['uploader_sub_count'] ** 2

log_uploader_sub_count = np.log(dislikes['uploader_sub_count'])
dislikes['log_uploader_sub_count^2'] = log_uploader_sub_count ** 2

dislikes['sub_count*view_count'] = dislikes['uploader_sub_count'] * dislikes['view_count']

# Best Subset Selection
y = dislikes['dislike_count'] # Response vector
X = dislikes.drop('dislike_count', axis=1) # Explanatory Matrix
X = sm.add_constant(X)

# Courtesy of https://www.science.smith.edu/~jcrouser/SDS293/labs/lab8-py.html
def process_subset(col_names):
    # Fit model on feature set and calculate RSS and BIC
    subset = X[list(col_names)].astype(int)

    model = sm.OLS(y, subset)
    regr = model.fit()

    RSS = regr.ssr
    BIC = regr.bic

    return {"model": model, "RSS": RSS, "BIC": BIC}
def best_subset_selection(k):
    results = []

    for subset in combinations(list(X), k):
        results.append(process_subset(subset))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the smallest RSS
    best_model = models.loc[models['RSS'].argmin()]

    # Return the best model, along with some other useful information about the model
    return best_model

# Dataframe to store our models
best_models = pd.DataFrame(columns=["BIC", "RSS", "model"])

k = len(list(X))
for i in range(1, k):
    best_models.loc[i] = best_subset_selection(i)

min_bic_idx = best_models['BIC'].idxmin() # Find the smallest BIC among our "best models"
best_model = best_models.loc[min_bic_idx, "model"].fit()

best_params = best_model.params.index.tolist() # Choose "best" parameters
X = X[best_params] # Filter training set

lr = LinearRegression()
results = cross_val_score(lr, X, y, scoring='neg_root_mean_squared_error', cv=10)
rmse = -results.mean()

print(rmse)
