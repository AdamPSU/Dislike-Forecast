import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from copy import deepcopy 

# Load the train set
dislikes = pd.read_csv("/path/to/train.csv")
# Load the test set
test = pd.read_csv("/path/to/test.csv")

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
test_sub = deepcopy(test) # To ensure original copy remains unchanged
test_sub = test_sub.drop('dislike_count', axis=1)

dislikes[dislikes['uploader_sub_count'] < 0] = 0 

# Reciprocate the sub count 
dislikes['uploader_sub_count_log'] = np.log1p(dislikes['uploader_sub_count'])
dislikes = dislikes.drop('uploader_sub_count', axis=1)

test_sub[test_sub['uploader_sub_count'] < 0] = 0 

# Reciprocate the sub count 
test_sub['uploader_sub_count_log'] = np.log1p(test_sub['uploader_sub_count'])
test_sub = test_sub.drop('uploader_sub_count', axis=1)

# Encoding
cat = ['has_subtitles', 'is_comments_enabled', 'is_ads_enabled', 'is_live_content',
       'is_age_limit']

dislikes = pd.get_dummies(dislikes, columns=cat, drop_first=True)
test_sub = pd.get_dummies(test_sub, columns=cat, drop_first=True)

# Prepare train/test data 
X = dislikes.drop('dislike_count', axis=1)
y = dislikes['dislike_count']

# Cross Validation & Model Testing

lr = LinearRegression() 
linear_cv = cross_val_score(lr, X, y, cv=10, scoring='neg_root_mean_squared_error')
rmse = -linear_cv.mean() 

print(f'With linear regression, our rmse is {round(rmse, 4)}.') # 406.0525

alphas = [0.01, 0.1, 1, 10, 100]
param_grid = {'alpha': alphas}

# ------------------------------------------------------------------ 

lasso = Lasso()

# Set up Grid Search to find best alpha value
lasso_cv = GridSearchCV(lasso, param_grid, cv=10, scoring='neg_root_mean_squared_error')
lasso_cv.fit(X, y)

# Get the best parameters and rmse
lasso_best_lambda = lasso_cv.best_params_['alpha'] # 0.1
rmse = -lasso_cv.best_score_  # Note the negative sign to convert back to RMSE

print(f'With LASSO, the rmse is {round(rmse, 4)} for a lambda of {lasso_best_lambda}.') # 406.0514, 0.1

# ------------------------------------------------------------------

ridge = Ridge() 

ridge_cv = GridSearchCV(ridge, param_grid, cv=10, scoring='neg_root_mean_squared_error')
ridge_cv.fit(X, y)

# Get the best parameters and rmse
ridge_best_lambda = ridge_cv.best_params_['alpha'] # 0.1
rmse = -ridge_cv.best_score_  # Note the negative sign to convert back to RMSE

print(f'With ridge, the rmse is {round(rmse, 4)} for a lambda of {ridge_best_lambda}.') # 406.0524, 100

# Fitting & Residual Analysis 
lr.fit(X, y) # I've chosen lr since the shrinkage methods don't seem to have strong impact
y_pred = lr.predict(X)
resid = y_pred - y


# Scatter plot of residuals vs fitted values
sns.scatterplot(x=y_pred, y=resid)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted', fontweight='bold')
 
plt.show() # Linear Regression might not be the best model for this job

# Predictions
test_pred = lr.predict(test_sub)
test['prediction'] = test_pred 

final = test[['dislike_count', 'prediction']]

