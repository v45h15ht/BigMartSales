import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sbn
import optuna
import plotly.graph_objects as go
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings('ignore')  # ignore notifications

df = pd.read_csv('data/train_v9rqX0R.csv')
df_type_weight = df[['Item_Type', 'Item_Weight']].groupby('Item_Type').mean().reset_index()
df1 = pd.merge(df, df_type_weight, on='Item_Type', how='left', suffixes=('', '_mean'))
df2 = df1[df1['Item_Weight'].isna()].copy()
df1 = df1[~df1['Item_Weight'].isna()]
df1.drop(columns=['Item_Weight_mean'], axis=1, inplace=True)
df2.drop(columns=['Item_Weight'], axis=1, inplace=True)
df2.rename(columns={'Item_Weight_mean': 'Item_Weight'}, inplace=True)
df3 = pd.concat([df1, df2], axis=0).copy()
df3['Outlet_Size'] = df3['Outlet_Size'].fillna(value='Medium')
df3.drop(columns=['Item_Identifier'], axis=1, inplace=True)
df4 = pd.get_dummies(df3)
features = list(df4.columns)
features.remove('Item_Outlet_Sales')
X = df4[features]
y = df4['Item_Outlet_Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

best_param = {
        'tree_method':'gpu_hist',
        'sampling_method': 'gradient_based',
        'lambda': 16.7305843317105,
        'alpha': 10.068697521317869,
        'eta': 1,
        'gamma': 20,
        'learning_rate': 0.008,
        'colsample_bytree': 1.0,
        'colsample_bynode': 0.6,
        'n_estimators':623,
        'min_child_weight': 193,
        'max_depth': 4,  
        'subsample': 1.0,
        'random_state': 42
}

best_model = XGBRegressor(**best_param)  
best_model.fit(X_train, y_train)

y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

res = pd.read_csv('data/test_AbJTz2l.csv')
res_type_weight = res[['Item_Type', 'Item_Weight']].groupby('Item_Type').mean().reset_index()
res1 = pd.merge(res, res_type_weight, on='Item_Type', how='left', suffixes=('', '_mean'))
res2 = res1[res1['Item_Weight'].isna()].copy()
res1 = res1[~res1['Item_Weight'].isna()]
res1.drop(columns=['Item_Weight_mean'], axis=1, inplace=True)
res2.drop(columns=['Item_Weight'], axis=1, inplace=True)
res2.rename(columns={'Item_Weight_mean': 'Item_Weight'}, inplace=True)
res3 = pd.concat([res1, res2], axis=0).copy()
res3['Outlet_Size'] = res3['Outlet_Size'].fillna(value='Medium')
res_identifier = res3[['Item_Identifier', 'Outlet_Identifier']].copy()
res3.drop(columns=['Item_Identifier'], axis=1, inplace=True)
res4 = pd.get_dummies(res3)
features = list(df4.columns)
features.remove('Item_Outlet_Sales')
features_test = list(res4.columns)

X_sub = res4[features]
y_sub = best_model.predict(X_sub)

df_sub = res_identifier.copy()
y_sub = pd.DataFrame(y_sub, columns=['Item_Outlet_Sales'])

df_final = pd.merge(df_sub, y_sub, left_index=True, right_index=True)
df_final.to_csv('submission.csv')