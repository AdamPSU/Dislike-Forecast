<div align="center">
  <h1>YouTube Dislike Predictor</h1>
  <a href="https://ibb.co/D5ZXCxw">
    <img src="https://i.ibb.co/5BPD8Ns/youtube.png" alt="youtube" border="0">
  </a>
</div>

## Overview
This project aims to predict the number of dislikes on YouTube videos using various regression techniques, including Linear Regression, Lasso, and Ridge Regression. The project involves several steps, including Exploratory Data Analysis (EDA), feature engineering, encoding, cross-validation, model evaluation, and residual analysis.

## Table of Contents
- [Overview](#overview)
- [Kaggle Notebook](#kaggle-notebook)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Encoding](#encoding)
- [Modeling](#modeling)
  - [Linear Regression](#linear-regression)
  - [Lasso Regression](#lasso-regression)
  - [Ridge Regression](#ridge-regression)
- [Cross-Validation](#cross-validation)
- [Residual Analysis](#residual-analysis)
- [Conclusion](#conclusion)

## Kaggle Notebook
You can find the complete notebook for this project on Kaggle [here](https://www.kaggle.com/code/adampsu/dislike-forecast).

## Data Description
The dataset includes various features of YouTube videos, such as upload date, uploader information, view count, like count, and more. The target variable is `dislike_count`. 

## Exploratory Data Analysis (EDA)
During the EDA phase, we checked for missing values and explored relationships between features and the target variable. Key insights from EDA helped guide feature engineering and model selection.

## Feature Engineering
Feature engineering involved several steps:
1. **Datetime Conversion**: Converted `upload_date` to datetime format and calculated the age of the video in days.
2. **Log Transformation**: Applied log transformation to `uploader_sub_count` to handle skewness.
3. **Dropping Irrelevant Features**: Removed features such as `upload_date`, `title`, and `description`.

## Encoding
Categorical variables were encoded using one-hot encoding to convert them into a format suitable for regression models. The variables encoded include:
- `has_subtitles`
- `is_comments_enabled`
- `is_ads_enabled`
- `is_live_content`
- `is_age_limit`

## Modeling
### Linear Regression
A simple linear regression model was trained using the processed features. Cross-validation was performed to evaluate the model's performance.

### Lasso Regression
Lasso regression was used to introduce regularization to the model, helping to prevent overfitting. Grid search was used to find the best alpha value for the Lasso model.

### Ridge Regression
Ridge regression, another regularization technique, was also employed. Grid search was used to find the optimal alpha value for the Ridge model.

## Cross-Validation
Cross-validation was used to evaluate the performance of each model. The Root Mean Squared Error (RMSE) was the primary evaluation metric.

## Residual Analysis
Residual analysis was conducted to assess the fit of the linear regression model. A scatter plot of residuals versus fitted values indicated potential issues with model fit.

## Conclusion
The project explored multiple regression techniques to predict YouTube dislikes. Linear regression, Lasso, and Ridge regression models were evaluated, with each model showing similar performance in terms of RMSE. Residual analysis suggested that linear regression might not be the best model for this task, indicating the need for further exploration of more advanced models or feature engineering techniques.
