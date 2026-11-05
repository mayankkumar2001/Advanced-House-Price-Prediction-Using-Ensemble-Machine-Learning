# Advanced House Price Prediction Using Ensemble Machine Learning

## Project Overview
This project develops a regression workflow to estimate residential property sale prices using property characteristics such as quality, area, construction details, rooms, bathrooms, basement, garage, and neighbourhood-related attributes.

## Objectives
- Explore relationships between property features and sale prices.
- Investigate outliers and missing values.
- Engineer meaningful features from property attributes.
- Prepare numerical and categorical variables for modelling.
- Train, tune, and compare regression and ensemble models.

## Dataset
The notebook uses `train.csv` containing the `SalePrice` target and `test.csv` for prediction. The `Id` column is retained to associate predictions with test records.

## Project Workflow
1. Explore the training and test datasets and inspect feature relationships.
2. Investigate missing values and selected outliers.
3. Engineer property-related features.
4. Impute, encode, and scale features using preprocessing pipelines.
5. Train, tune, and compare regression and ensemble models.
6. Generate predictions and export the submission file.

## Machine Learning Models
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Ridge Regression
- Gradient Boosting Regressor
- LightGBM Regressor
- CatBoost Regressor
- Voting Regressor
- Stacking Regressor

## Evaluation
The notebook uses Root Mean Squared Error (RMSE) to compare regression models and creates `submission.csv` using the stacking prediction workflow.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- SciPy
- LightGBM
- CatBoost

## Project Notebook
Open the project notebook: `Pro(4).ipynb`.

## Complete Project Code
The complete source code from the notebook is included below. Expand the section to view it.

<details>
<summary>View complete project code</summary>

### Code Cell 1

```python
# !pip install lightgbm
```


### Code Cell 2

```python
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
```


### Code Cell 3

```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```


### Code Cell 4

```python
test_df.head()
```


### Code Cell 5

```python
train_df.head()
```


### Code Cell 6

```python
train_df.describe()
```


### Code Cell 7

```python
train_df.select_dtypes(exclude=['object', 'str']).dtypes
```


### Code Cell 8

```python
plt.scatter(x='MSSubClass',y='SalePrice',data=train_df)
```


### Code Cell 9

```python
plt.scatter(x='LotFrontage',y='SalePrice',data=train_df)
```


### Code Cell 10

```python
train_df.query('LotFrontage > 300')
## Drop 935, 1299
```


### Code Cell 11

```python
plt.scatter(x='LotArea',y='SalePrice',data=train_df)
```


### Code Cell 12

```python
train_df.query('LotArea >55000')
```


### Code Cell 13

```python
#250 ,#314 ,# 336 ,#707
# 1397
```


### Code Cell 14

```python
plt.scatter(x='OverallQual',y='SalePrice',data=train_df)
```


### Code Cell 15

```python
train_df.query('OverallQual == 10')
#524
```


### Code Cell 16

```python
plt.scatter(x='OverallCond',y='SalePrice',data=train_df)
plt.show()
```


### Code Cell 17

```python
train_df.query('OverallCond ==2')
# drop 379
```


### Code Cell 18

```python
train_df.query('OverallCond == 5 & SalePrice > 700000')
#1183
```


### Code Cell 19

```python
train_df.query('OverallCond == 6 & SalePrice > 700000')
#692
```


### Code Cell 20

```python
plt.scatter(x='YearBuilt',y='SalePrice',data=train_df)
```


### Code Cell 21

```python
train_df.query('YearBuilt < 1900 & SalePrice > 400000')
# drop 186
```


### Code Cell 22

```python
train_df.query('YearBuilt > 1980 & SalePrice > 700000')
#186
```


### Code Cell 23

```python
plt.scatter(x='YearRemodAdd',y='SalePrice',data=train_df)
```


### Code Cell 24

```python
train_df.query('YearRemodAdd < 1970 & SalePrice>300000')
#314
```


### Code Cell 25

```python
plt.scatter(x='MasVnrArea',y='SalePrice',data=train_df)
```


### Code Cell 26

```python
train_df.query('MasVnrArea > 1500')
#298
```


### Code Cell 27

```python
plt.scatter(x='BsmtFinSF1',y='SalePrice',data=train_df)
```


### Code Cell 28

```python
train_df.query('BsmtFinSF1 >5000')
#1299
```


### Code Cell 29

```python
plt.scatter(x='BsmtFinSF2',y='SalePrice',data=train_df)
```


### Code Cell 30

```python
train_df.query('BsmtFinSF2 > 400 & SalePrice > 500000')
#441
```


### Code Cell 31

```python
plt.scatter(x='BsmtUnfSF',y='SalePrice',data=train_df)
```


### Code Cell 32

```python
plt.scatter(x='TotalBsmtSF',y='SalePrice',data=train_df)
```


### Code Cell 33

```python
train_df.query('TotalBsmtSF > 5000')
#1299
```


### Code Cell 34

```python
plt.scatter(x='1stFlrSF',y='SalePrice',data=train_df)
```


### Code Cell 35

```python
train_df[train_df['1stFlrSF'] > 4000]#1299
```


### Code Cell 36

```python
plt.scatter(x='2ndFlrSF',y='SalePrice',data=train_df)
```


### Code Cell 37

```python
train_df[train_df['2ndFlrSF'] > 2000]
```


### Code Cell 38

```python
plt.scatter(x='LowQualFinSF',y='SalePrice',data=train_df)
```


### Code Cell 39

```python
plt.scatter(x='GrLivArea',y='SalePrice',data=train_df)
```


### Code Cell 40

```python
train_df.query('GrLivArea > 4300')
#524 1299
```


### Code Cell 41

```python
plt.scatter(x='BsmtFullBath',y='SalePrice',data=train_df)
```


### Code Cell 42

```python
train_df.query('BsmtFullBath == 3')
#739
```


### Code Cell 43

```python
plt.scatter(x='BsmtHalfBath',y='SalePrice',data=train_df)
```


### Code Cell 44

```python
np.unique(stats.zscore(train_df['BsmtHalfBath'].dropna()))
```


### Code Cell 45

```python
train_df.query('BsmtHalfBath == 2')
#598 955
```


### Code Cell 46

```python
plt.scatter(x='FullBath',y='SalePrice',data=train_df)
```


### Code Cell 47

```python
plt.scatter(x='HalfBath',y='SalePrice',data=train_df)
```


### Code Cell 48

```python
plt.scatter(x='BedroomAbvGr',y='SalePrice',data=train_df)
```


### Code Cell 49

```python
train_df.query('BedroomAbvGr == 8')
#636
```


### Code Cell 50

```python
plt.scatter(x='KitchenAbvGr',y='SalePrice',data=train_df)
```


### Code Cell 51

```python
train_df.query('KitchenAbvGr == 3')
#49 810
```


### Code Cell 52

```python
plt.scatter(x='TotRmsAbvGrd', y='SalePrice', data=train_df)
```


### Code Cell 53

```python
train_df.query('TotRmsAbvGrd == 14')
#636
```


### Code Cell 54

```python
plt.scatter(x='Fireplaces', y='SalePrice', data=train_df)
```


### Code Cell 55

```python
plt.scatter(x='GarageYrBlt', y='SalePrice', data=train_df)
```


### Code Cell 56

```python
plt.scatter(x='GarageArea', y='SalePrice', data=train_df)
```


### Code Cell 57

```python
train_df.query('GarageArea > 1200')
#1062, 1191
```


### Code Cell 58

```python
plt.scatter(x='WoodDeckSF', y='SalePrice', data=train_df)
```


### Code Cell 59

```python
plt.scatter(x='OpenPorchSF', y='SalePrice', data=train_df)
```


### Code Cell 60

```python
train_df.query('OpenPorchSF > 500')
#496
```


### Code Cell 61

```python
plt.scatter(x='EnclosedPorch', y='SalePrice', data=train_df)
```


### Code Cell 62

```python
train_df.query('EnclosedPorch > 500')
#198
```


### Code Cell 63

```python
plt.scatter(x='3SsnPorch', y='SalePrice', data=train_df)
```


### Code Cell 64

```python
plt.scatter(x='ScreenPorch', y='SalePrice', data=train_df)
```


### Code Cell 65

```python
plt.scatter(x='PoolArea', y='SalePrice', data=train_df)
```


### Code Cell 66

```python
values = [598, 955, 935, 1299, 250, 314, 336, 707, 379, 1183, 692, 186, 441, 186, 524, 739, 598, 955, 636, 1062, 1191, 496, 198, 1338]
```


### Code Cell 67

```python
train_df = train_df[train_df.Id.isin(values) == False]
```


### Code Cell 68

```python
train_df
```


### Code Cell 69

```python
pd.DataFrame(train_df.isnull().sum().sort_values(ascending=False)).head(20)
```


### Code Cell 70

```python
train_df['PoolQC'].unique()
```


### Code Cell 71

```python
train_df['MiscFeature'].unique()
```


### Code Cell 72

```python
train_df['Alley'].unique()
```


### Code Cell 73

```python
train_df['Alley']= train_df['Alley'].fillna('No')
test_df['Alley']= train_df['Alley'].fillna('No')
```


### Code Cell 74

```python
sns.boxplot(data=train_df, x="Alley", y="SalePrice")
```


### Code Cell 75

```python
train_df.query('Alley == "Grvl"').count()
```


### Code Cell 76

```python
train_df.query('Alley == "Pave"').count()
```


### Code Cell 77

```python
train_df['Fence'].unique()
```


### Code Cell 78

```python
train_df['Fence'] = train_df['Fence'].fillna('No')
```


### Code Cell 79

```python
test_df['Fence'] = train_df['Fence'].fillna('No')
```


### Code Cell 80

```python
sns.boxplot(x='Fence',y='SalePrice',data=train_df)
```


### Code Cell 81

```python
train_df['MasVnrType'].unique()
```


### Code Cell 82

```python
train_df['MasVnrType'] = train_df['Fence'].fillna('No')
```


### Code Cell 83

```python
test_df['MasVnrType'] = train_df['Fence'].fillna('No')
```


### Code Cell 84

```python
sns.boxplot(x='MasVnrType',y='SalePrice',data=train_df)
```


### Code Cell 85

```python
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
```


### Code Cell 86

```python
train_df['FireplaceQu'].unique()
```


### Code Cell 87

```python
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('No')
train_df['FireplaceQu'] = train_df['FireplaceQu'].fillna('No')
```


### Code Cell 88

```python
sns.boxplot(x='FireplaceQu',y='SalePrice',data=train_df)
```


### Code Cell 89

```python
sns.boxplot(x='Fireplaces',y='SalePrice',data=train_df)
```


### Code Cell 90

```python
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(0)
```


### Code Cell 91

```python
train_df['GarageYrBlt'].corr(train_df['YearBuilt'])
```


### Code Cell 92

```python
### Since the cor relation is high with of GarageYrBlt and YearBuilt we will drop this
```


### Code Cell 93

```python
#train_df.drop(columns='GarageYrBlt',inplace=True)
```


### Code Cell 94

```python
train_df['GarageCond'].unique()
```


### Code Cell 95

```python
train_df['GarageCond'] = train_df['GarageCond'].fillna('no')
train_df['GarageCond'] = train_df['GarageCond'].fillna('no')
```


### Code Cell 96

```python
sns.boxplot(x='GarageCond',y='SalePrice',data=train_df)
```


### Code Cell 97

```python
train_df['GarageType'].unique()
```


### Code Cell 98

```python
train_df['GarageType'] = train_df['GarageType'].fillna('No')
train_df['GarageType'] = train_df['GarageType'].fillna('No')
```


### Code Cell 99

```python
sns.boxplot(x='GarageType',y='SalePrice',data=train_df)
```


### Code Cell 100

```python
train_df['GarageFinish'].unique()
```


### Code Cell 101

```python
train_df['GarageFinish'] = train_df['GarageFinish'].fillna('No')
train_df['GarageFinish'] = train_df['GarageFinish'].fillna('No')
```


### Code Cell 102

```python
sns.boxplot(x='GarageFinish',y='SalePrice',data=train_df)
```


### Code Cell 103

```python
train_df['GarageQual'].unique()
```


### Code Cell 104

```python
train_df['GarageQual'] = train_df['GarageQual'].fillna('No')
train_df['GarageQual'] = train_df['GarageQual'].fillna('No')
```


### Code Cell 105

```python
sns.boxplot(x='GarageQual',y='SalePrice',data=train_df)
```


### Code Cell 106

```python
train_df['BsmtFinType2'].unique()
```


### Code Cell 107

```python
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna('Unf')
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna('Unf')
```


### Code Cell 108

```python
sns.boxplot(x='BsmtFinType2',y='SalePrice',data=train_df)
```


### Code Cell 109

```python
train_df['BsmtExposure'].unique()
```


### Code Cell 110

```python
train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna('No')
```


### Code Cell 111

```python
test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna('No')
```


### Code Cell 112

```python
train_df['BsmtQual'].unique()
```


### Code Cell 113

```python
train_df['BsmtQual'] = train_df['BsmtQual'].fillna('No')
train_df['BsmtQual'] = train_df['BsmtQual'].fillna('No')
```


### Code Cell 114

```python
sns.boxplot(x='BsmtQual',y='SalePrice',data=train_df)
```


### Code Cell 115

```python
train_df['BsmtFinType1'].unique()
```


### Code Cell 116

```python
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna('Unf')
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna('Unf')
```


### Code Cell 117

```python
sns.boxplot(x='BsmtFinType1',y='SalePrice',data=train_df)
```


### Code Cell 118

```python
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(0)
```


### Code Cell 119

```python
train_df['Electrical'].unique()
```


### Code Cell 120

```python
train_df['Electrical'] = train_df['Electrical'].fillna('SBrkr')
train_df['Electrical'] = train_df['Electrical'].fillna('SBrkr')
```


### Code Cell 121

```python
train_df = train_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageCond', 'BsmtFinType2'])
test_df = test_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'GarageYrBlt', 'GarageCond', 'BsmtFinType2'])
```


### Code Cell 122

```python
train_df['houseage'] = train_df['YrSold'] - train_df['YearBuilt']
test_df['houseage'] = test_df['YrSold'] - test_df['YearBuilt']
```


### Code Cell 123

```python
train_df['houseremodelage'] = train_df['YrSold'] - train_df['YearRemodAdd']
test_df['houseremodelage'] = test_df['YrSold'] - test_df['YearRemodAdd']
```


### Code Cell 124

```python
train_df['totalsf'] = train_df['1stFlrSF'] + train_df['2ndFlrSF'] + train_df['BsmtFinSF1'] + train_df['BsmtFinSF2']
test_df['totalsf'] = test_df['1stFlrSF'] + test_df['2ndFlrSF'] + test_df['BsmtFinSF1'] + test_df['BsmtFinSF2']
```


### Code Cell 125

```python
train_df['totalarea'] = train_df['GrLivArea'] + train_df['TotalBsmtSF']
test_df['totalarea'] = test_df['GrLivArea'] + test_df['TotalBsmtSF']
```


### Code Cell 126

```python
train_df['totalbaths'] = train_df['BsmtFullBath'] + train_df['FullBath'] + 0.5 * (train_df['BsmtHalfBath'] + train_df['HalfBath']) 
test_df['totalbaths'] = test_df['BsmtFullBath'] + test_df['FullBath'] + 0.5 * (test_df['BsmtHalfBath'] + test_df['HalfBath'])
```


### Code Cell 127

```python
train_df['totalporchsf'] = train_df['OpenPorchSF'] + train_df['3SsnPorch'] + train_df['EnclosedPorch'] + train_df['ScreenPorch'] + train_df['WoodDeckSF']
test_df['totalporchsf'] = test_df['OpenPorchSF'] + test_df['3SsnPorch'] + test_df['EnclosedPorch'] + test_df['ScreenPorch'] + test_df['WoodDeckSF']
```


### Code Cell 128

```python
train_df = train_df.drop(columns=['Id','YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'TotalBsmtSF','BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'])
test_df = test_df.drop(columns=['YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'TotalBsmtSF','BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 'ScreenPorch','WoodDeckSF'])
```


### Code Cell 129

```python
plt.figure(figsize=(20,12))
sns.heatmap(train_df.corr(numeric_only=True),annot=True,cmap='coolwarm',fmt=".2f")
```


### Code Cell 130

```python
#drop GarageArea since they are very highly corerelated
```


### Code Cell 131

```python
train_df.drop(columns=['GarageArea'],inplace=True)
test_df.drop(columns=['GarageArea'],inplace=True)
```


### Code Cell 132

```python
sns.histplot(
    train_df,
    x=train_df['SalePrice']
)
```


### Code Cell 133

```python
train_df['SalePrice'] = np.log1p(train_df['SalePrice'])
```


### Code Cell 134

```python
train_df.head()
```


### Code Cell 135

```python
sns.histplot(
    train_df,
    x=train_df['SalePrice']
)
```


### Code Cell 136

```python
train_df.dtypes[train_df.dtypes == 'str']
```


### Code Cell 137

```python
train_df.dtypes[train_df.dtypes != 'str']
```


### Code Cell 138

```python
ode_cols = ['LotShape', 'LandContour','Utilities','LandSlope',  'BsmtQual',  'BsmtFinType1',  'CentralAir',  'Functional', \
           'FireplaceQu', 'GarageFinish', 'GarageQual', 'PavedDrive', 'ExterCond', 'KitchenQual', 'BsmtExposure', 'HeatingQC','ExterQual', 'BsmtCond']
```


### Code Cell 139

```python
ohe_cols = ['Street', 'LotConfig','Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', \
           'MasVnrType','Foundation',  'Electrical',  'SaleType', 'MSZoning', 'SaleCondition', 'Heating', 'GarageType', 'RoofMatl']
```


### Code Cell 140

```python
num_cols = train_df.select_dtypes(include=['int64','float64']).columns
```


### Code Cell 141

```python
num_cols = num_cols.drop('SalePrice')
```


### Code Cell 142

```python
num_pipeline = Pipeline(steps=[
    ('impute',SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
```


### Code Cell 143

```python
ode_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])
```


### Code Cell 144

```python
ohe_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
```


### Code Cell 145

```python
col_trans = ColumnTransformer(transformers=[
    ('num_p', num_pipeline, num_cols),
    ('ode_p', ode_pipeline, ode_cols),
    ('ohe_p', ohe_pipeline, ohe_cols),
    ],
    remainder='passthrough', 
    n_jobs=-1)
```


### Code Cell 146

```python
pipeline = Pipeline(steps=[
    ('preprocessing', col_trans)
])
```


### Code Cell 147

```python
X = train_df.drop('SalePrice',axis=1)
y = train_df['SalePrice']
```


### Code Cell 148

```python
X_preprocessed = pipeline.fit_transform(X)
```


### Code Cell 149

```python
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=25)
```


### Code Cell 150

```python
### BUILD models
```


### Code Cell 151

```python
lr = LinearRegression()
```


### Code Cell 152

```python
lr.fit(X_train,y_train)
```


### Code Cell 153

```python
y_pred_lr = lr.predict(X_test)
```


### Code Cell 154

```python
mean_squared_error(y_test,y_pred_lr)
```


### Code Cell 155

```python
RFR = RandomForestRegressor(random_state=13)
```


### Code Cell 156

```python
param_grid_RFR = {
    'max_depth': [5, 10, 15],
    'n_estimators': [100, 250, 500],
    'min_samples_split': [3, 5, 10]
}
```


### Code Cell 157

```python
rfr_cv = GridSearchCV(RFR, param_grid_RFR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
```


### Code Cell 158

```python
rfr_cv.fit(X_train,y_train)
```


### Code Cell 159

```python
np.sqrt(-1 * rfr_cv.best_score_)
```


### Code Cell 160

```python
rfr_cv.best_params_
```


### Code Cell 161

```python
XGB = XGBRegressor(random_state = 13)
```


### Code Cell 162

```python
param_grid_XGB = {
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [300],
    'max_depth': [3],
    'min_child_weight': [1,2,3],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
}
```


### Code Cell 163

```python
xgb_cv = GridSearchCV(XGB, param_grid_XGB, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
```


### Code Cell 164

```python
xgb_cv.fit(X_train, y_train)
```


### Code Cell 165

```python
np.sqrt(-1 * xgb_cv.best_score_)
```


### Code Cell 166

```python
param_grid_Ridge = {
    'alpha' : [0.05,0.1,1,3,5,10],
    'solver' : ['auto','svd','cholesky','lqsr','sparse_cg','sag']
}
```


### Code Cell 167

```python
ridge_cv = GridSearchCV(XGB, param_grid_XGB, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
```


### Code Cell 168

```python
ridge_cv.fit(X_train, y_train)
```


### Code Cell 169

```python
np.sqrt(-1 * ridge_cv.best_score_)
```


### Code Cell 170

```python
GBR = GradientBoostingRegressor()
```


### Code Cell 171

```python
param_grid_GBR = {
    'max_depth' : [12,15,20],
    'n_estimators' : [200,300,1000],
    'min_samples_leaf' : [10,25,50],
    'learning_rate' : [0.001,0.01,0.1],
    'max_features' : [0.01,0.1,0.7]
}
```


### Code Cell 172

```python
GBR_cv = GridSearchCV(GBR, param_grid_GBR, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
```


### Code Cell 173

```python
GBR_cv.fit(X_train, y_train)
```


### Code Cell 174

```python
np.sqrt(-1 * GBR_cv.best_score_)
```


### Code Cell 175

```python
lgbm_regressor = lgb.LGBMRegressor()
```


### Code Cell 176

```python
param_grid_LGBM = {
    'boosting_type' : ['gbdt','dart'],
    'num_leaves' : [20,30,40],
    'learning_rate' : [0.01,0.05,0.1],
    'n_estimators' : [100,200,300]
}
```


### Code Cell 177

```python
lgbm_cv = GridSearchCV(GBR, param_grid_GBR, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
```


### Code Cell 178

```python
lgbm_cv.fit(X_train,y_train)
```


### Code Cell 179

```python
np.sqrt(-1 * lgbm_cv.best_score_)
```


### Code Cell 180

```python
catboost = CatBoostRegressor(loss_function='RMSE', verbose=False)
```


### Code Cell 181

```python
param_grid_cat ={
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.5]
}
```


### Code Cell 182

```python
cat_cv = GridSearchCV(catboost, param_grid_cat, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
```


### Code Cell 183

```python
cat_cv.fit(X_train, y_train)
```


### Code Cell 184

```python
np.sqrt(-1 * cat_cv.best_score_)
```


### Code Cell 185

```python
vr = VotingRegressor([('gbr', GBR_cv.best_estimator_),
                      ('xgb', xgb_cv.best_estimator_),
                      ('ridge', ridge_cv.best_estimator_)],
                    weights=[2,3,1])
```


### Code Cell 186

```python
vr.fit(X_train, y_train)
```


### Code Cell 187

```python
y_pred_vr = vr.predict(X_test)
```


### Code Cell 188

```python
np.sqrt(mean_squared_error(y_test, y_pred_vr))
```


### Code Cell 189

```python
estimators = [
    ('gbr', GBR_cv.best_estimator_),
    ('xgb', xgb_cv.best_estimator_),
    ('cat', cat_cv.best_estimator_),
    ('lgb', lgbm_cv.best_estimator_),
    ('rfr', rfr_cv.best_estimator_),
]
```


### Code Cell 190

```python
stackreg = StackingRegressor(
            estimators = estimators,
            final_estimator = vr
)
```


### Code Cell 191

```python
stackreg.fit(X_train, y_train)
```


### Code Cell 192

```python
y_pred_stack = stackreg.predict(X_test)
```


### Code Cell 193

```python
np.sqrt(mean_squared_error(y_test, y_pred_stack ))
```


### Code Cell 194

```python
df_test_preprocess = pipeline.transform(test_df)
```


### Code Cell 195

```python
models = {
    'Linear Regression': lr,
    'Random Forest': rfr_cv.best_estimator_,
    'XGBoost': xgb_cv.best_estimator_,
    'Ridge': ridge_cv.best_estimator_,
    'Gradient Boosting': GBR_cv.best_estimator_,
    'LightGBM': lgbm_cv.best_estimator_,
    'CatBoost': cat_cv.best_estimator_,
    'Voting Regressor': vr,
    'Stacking Regressor': stackreg
}

model_comparison = pd.DataFrame({
    'Model': name,
    'RMSE': np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
} for name, model in models.items())

model_comparison.sort_values(by='RMSE').reset_index(drop=True)
```


</details>

## Resume Project Title
**Advanced House Price Prediction Using Ensemble Machine Learning**

**Resume description:** Developed a house price prediction pipeline involving outlier analysis, missing-value imputation, feature engineering, categorical encoding, scaling, and hyperparameter tuning; compared regression and ensemble models using RMSE and generated test-set predictions.
