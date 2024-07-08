import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from itertools import combinations
from copy import deepcopy

dislikes = pd.read_csv('../data/raw/train.csv')
test = pd.read_csv("../data/raw/test.csv")

# Convert to pandas datetime to access additional attributes
dislikes['upload_date'] = pd.to_datetime(dislikes['upload_date'], errors='coerce')
# Obtain the number of days
dislikes['age'] = (pd.Timestamp.today() - dislikes["upload_date"]).dt.days

dislikes = dislikes.drop(dislikes[['upload_date', 'title', 'description']], axis=1)
dislikes = dislikes.dropna()

# Convert to pandas datetime to access additional attributes
test['upload_date'] = pd.to_datetime(test['upload_date'], errors='coerce')
# Obtain the number of days
test['age'] = (pd.Timestamp.today() - test["upload_date"]).dt.days

test = test.drop(test[['upload_date', 'title', 'description']], axis=1)
test = test.dropna()

test_sub = deepcopy(test)

# Square the sub count
dislikes['uploader_sub_count^2'] = dislikes['uploader_sub_count'] ** 2
# Square the like count
dislikes['like_count^2'] = dislikes['like_count'] ** 2
# Create an interaction term between sub count and view count
dislikes['sub_count*view_count'] = dislikes['uploader_sub_count'] * dislikes['view_count']

# Square the sub count
test_sub['uploader_sub_count^2'] = test_sub['uploader_sub_count'] ** 2
# Square the like count
test_sub['like_count^2'] = test_sub['like_count'] ** 2
# Create an interaction term between sub count and view count
test_sub['sub_count*view_count'] = test_sub['uploader_sub_count'] * test_sub['view_count']

# Encoding
cat = ['has_subtitles', 'is_comments_enabled', 'is_ads_enabled']

dislikes = pd.get_dummies(dislikes, columns=cat, drop_first=True)
test_sub = pd.get_dummies(test_sub, columns=cat, drop_first=True)

# Split into train/test
y = dislikes['dislike_count'] # Response vector
X = dislikes.drop('dislike_count', axis=1) # Explanatory Matrix
X = sm.add_constant(X)

# Best Subset Selection
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
test_sub = test_sub[best_params] # Filter test set

lr = LinearRegression()
lr_results = cross_val_score(lr, X, y, scoring='neg_root_mean_squared_error', cv=10)
lr_rmse = -lr_results.mean()
print(f"CV RMSE: {lr_rmse}") # 452.556

# Fit the model
lr.fit(X, y)

# Make predictions
y_pred = lr.predict(test_sub)
y_pred[y_pred < 0] = 0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

print(f"TRUE RMSE: {rmse(test['dislike_count'], y_pred)}") # 542.980
