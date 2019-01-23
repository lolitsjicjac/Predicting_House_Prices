# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
author : Alberto Castillo
"""
#Find the price of house sales for the test set. 
#have the sample set to check our RMSE or our residuals 
#Using Advanced regression techniques like random forest and gradient boosting


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import featuretools as ft
import utils
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, median_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import seaborn as sns
from sklearn.preprocessing import scale, robust_scale, RobustScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge

train_houses = "/Users/Alberto/Desktop/Housing Regression/all/train.csv"
test_houses = "/Users/Alberto/Desktop/Housing Regression/all/test.csv"
sample_houses = "/Users/Alberto/Desktop/Housing Regression/all/sample_submission.csv"

train = pd.read_csv(train_houses)
test = pd.read_csv(test_houses)
sample = pd.read_csv(sample_houses)

house_prices = train['SalePrice'].copy()
#train = train.fillna(0)
sample_test = sample.drop("Id", axis =1)

#Lets fix the missing values that are in our models.
#First lets work on fixing the missing values on Training
train_na = (train.isnull().sum()/len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending = False)[:30]
missing_data_train = pd.DataFrame({'Missing Ratio': train_na})
missing_data_train = missing_data_train['Missing Ratio'].astype('float')

#Graphing our data
plt.xticks(rotation = 90)
sns.barplot(x = missing_data_train.index, y = missing_data_train)
plt.xlabel("Features")
plt.ylabel("Missing Ratio")
plt.title("The Missing Ratio in the Training Data")

#Now lets fill in the missing values into our training data

#PoolQC
train["PoolQC"] = train["PoolQC"].fillna("None")
#MiscFeature
train["MiscFeature"] = train["MiscFeature"].fillna("None")
#Alley
train["Alley"] = train["Alley"].fillna("None")

#FEnce
train["Fence"] = train["Fence"].fillna("None")
#FireplaceQu
train["FireplaceQu"] = train["FireplaceQu"].fillna("None")
#LotFrontage
med = train["LotFrontage"].median()
train["LotFrontage"] = train["LotFrontage"].fillna(med)
# GarageType, GarageFinish, GarageQual, GarageCond
for col in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
    train[col] = train[col].fillna("None")
#GarageYrBlt
train["GarageYrBlt"] = train["GarageYrBlt"].fillna(0)
#BsmtFinType1, BsmtFinType2, BsmtExposure, BsmtQual, BsmtCond
for col in ("BsmtFinType1", "BsmtFinType2", "BsmtExposure", "BsmtQual", "BsmtCond"):
    train[col] = train[col].fillna(0)
#MasVnrArea
train["MasVnrArea"] = train["MasVnrArea"].fillna(0)
#MasVnrType
train["MasVnrType"] = train["MasVnrType"].fillna("None")
#Electrical
train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])

#WE need to remove outliers
plt.scatter(x = train['GrLivArea'], y = train["SalePrice"])
plt.ylabel("SalePrice")
plt.xlabel("GrLivArea")
plt.show()
#Delete the two outliers
out = train[(train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000)]
train = train.drop([523, 1298])

#Now need to drop the outliers

#Lets see if we can find anymore relationships or outliers what can be removes
numeric_features = train.dtypes[train.dtypes != "object"].index
numeric_features = numeric_features.drop("Id")
numeric_data = list(numeric_features)

#TO visualize all the data at once 
for i,c in zip(range(1,36), numeric_data):
    plt.figure(1, figsize= (20,10))
    plt.subplot(6,6,i)
    plt.scatter(train[c], train["SalePrice"])
    plt.xlabel(c)
    plt.ylabel("SalePrice")
    plt.subplots_adjust(hspace = .8, wspace = .8)
    plt.show()

#Now we have looked at the whole plots and picked specific ones we deem to have some outliers
def graph_scatter(df, col):
    plt.figure(1, figsize= (10,5))
    plt.scatter(df[col], train["SalePrice"])
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.show()

graph_scatter(train, numeric_data[1])
graph_scatter(train, numeric_data[2]) #No drop yet
graph_scatter(train, numeric_data[7]) #ok
graph_scatter(train, numeric_data[9]) # ddrop >1400
graph_scatter(train, numeric_data[29]) # drop > 500
graph_scatter(train, numeric_data[33]) # drop > 8000

#Deleting more outliers 
lotfrontage = train[train["LotFrontage"] > 300] #934
BsmtFinSF2 = train[train["BsmtFinSF2"] > 1400] #322
EnclosedPorch = train[train["EnclosedPorch"] > 500] #197
MiscVal = train[train["MiscVal"] > 8000] #346,1230
train = train.drop([197, 322, 346, 934, 1230])



#Now removing values that we are predicting
house_prices = train["SalePrice"].copy()
train = train.drop("SalePrice", axis = 1)
#log transform skewed numeric features
numeric_features = train.dtypes[train.dtypes != "object"].index
numeric_features = numeric_features.drop("Id")
train[numeric_features] = np.log1p(train[numeric_features])

es = ft.EntitySet("Houses")

es = es.entity_from_dataframe(entity_id = "HousingSet",
                              dataframe= train,
                              index = "Id",
                              variable_types = {"MSSubClass": ft.variable_types.Categorical,
                                      "MSZoning": ft.variable_types.Categorical,
                                                "Street": ft.variable_types.Categorical,
                                                "Alley": ft.variable_types.Categorical,
                                                "LotShape": ft.variable_types.Categorical,
                                                "LandContour": ft.variable_types.Categorical,
                                                "Utilities": ft.variable_types.Categorical,
                                                "LotConfig": ft.variable_types.Categorical,
                                                "LandSlope": ft.variable_types.Categorical,
                                                "Neighborhood": ft.variable_types.Categorical,
                                                "Condition1": ft.variable_types.Categorical,
                                                "Condition2": ft.variable_types.Categorical,
                                                "BldgType": ft.variable_types.Categorical,
                                                "HouseStyle": ft.variable_types.Categorical,
                                                "RoofStyle": ft.variable_types.Categorical,
                                                "RoofMatl": ft.variable_types.Categorical,
                                                "Exterior1st": ft.variable_types.Categorical,
                                                "Exterior2nd": ft.variable_types.Categorical,
                                                "MasVnrType": ft.variable_types.Categorical,
                                                "ExterQual": ft.variable_types.Categorical,
                                                "ExterCond": ft.variable_types.Categorical,
                                                "Foundation": ft.variable_types.Categorical,
                                                "BsmtQual": ft.variable_types.Categorical,
                                                "BsmtCond": ft.variable_types.Categorical,
                                                "BsmtExposure": ft.variable_types.Categorical,
                                                "BsmtFinType1": ft.variable_types.Categorical,
                                                "BsmtFinType2": ft.variable_types.Categorical,
                                                "Heating": ft.variable_types.Categorical,
                                                "HeatingQC": ft.variable_types.Categorical,
                                                "CentralAir": ft.variable_types.Categorical,
                                                "Electrical": ft.variable_types.Categorical,
                                                "KitchenQual": ft.variable_types.Categorical,
                                                "Functional": ft.variable_types.Categorical,
                                                "Fireplaces": ft.variable_types.Categorical,
                                                "FireplaceQu": ft.variable_types.Categorical,
                                                "GarageType": ft.variable_types.Categorical,
                                                "GarageFinish": ft.variable_types.Categorical,
                                                "GarageQual": ft.variable_types.Categorical,
                                                "GarageCond": ft.variable_types.Categorical,
                                                "PavedDrive": ft.variable_types.Categorical,
                                                "PoolQC": ft.variable_types.Categorical,
                                                "Fence": ft.variable_types.Categorical,
                                                "MiscFeature": ft.variable_types.Categorical,
                                                "MoSold": ft.variable_types.Categorical,
                                                "YrSold": ft.variable_types.Categorical,
                                                "SaleType": ft.variable_types.Categorical,
                                                "SaleCondition": ft.variable_types.Categorical})



#Now going to work on the test set and clean it up 
test_na = (test.isnull() .sum()/len(test)) *100
test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending = False)
missing_data_test = pd.DataFrame({'Missing Ratio': test_na})
missing_data_test = missing_data_test['Missing Ratio'].astype('float')

#Lets clean up the data
#PoolQC
test["PoolQC"] = test["PoolQC"].fillna("None")

#MiscFeature
test["MiscFeature"] = test["MiscFeature"].fillna("None")
#Alley
test["Alley"] = test["Alley"].fillna("None")
#Fence
test["Fence"] = test["Fence"].fillna("None")
#FireplaceQu
test["FireplaceQu"] = test["FireplaceQu"].fillna("None")
#LotFrontage
med1 = test["LotFrontage"].median()
test["LotFrontage"] = test["LotFrontage"].fillna(med1)
#GarageYrBlt, GarageCars, GarageArea
for col in ("GarageYrBlt", "GarageCars", "GarageArea"):
    test[col] = test[col].fillna(0)

#GarageCond, GarageQual, GarageFinish, GarageType
for col in ("GarageCond", "GarageQual", "GarageFinish", "GarageType"):
    test[col] = test[col].fillna("None")
#BsmtCond, BsmtExposure, BsmtQual, BsmtFinType1, BsmtFinType2
for col in ("BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType1", "BsmtFinType2"):
    test[col] = test[col].fillna("None")
#MasVnrType
test["MasVnrType"] = test["MasVnrType"].fillna("None")
#MasVnrArea
test["MasVnrArea"] = test["MasVnrArea"].fillna(0)
#MSZoning
test["MSZoning"] = test["MSZoning"].fillna(test["MSZoning"].mode()[0])
#BsmtFullBath, BsmtHalfBath, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF
for col in ("BsmtFullBath", "BsmtHalfBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"):
    test[col] = test[col].fillna(0)

#Utilities
test = test.drop(["Utilities"], axis =1 )
#Functional
test["Functional"] = test["Functional"].fillna("Typ")
#Exterior1st, Exterior2nd
test["Exterior1st"] = test["Exterior1st"].fillna(test["Exterior1st"].mode()[0])
test["Exterior2nd"] = test["Exterior2nd"].fillna(test["Exterior2nd"].mode()[0])
#SaleType
test["SaleType"] = test["SaleType"].fillna(test["SaleType"].mode()[0])
#KitchenQual
test["KitchenQual"] = test["KitchenQual"].fillna(test["KitchenQual"].mode()[0])

#Fixing skewed values
test_features = test.dtypes[test.dtypes != "object"].index
test_features = test_features.drop("Id")
test[test_features] = np.log1p(test[test_features])
   

es = es.entity_from_dataframe(entity_id = "HousingTest",
                              dataframe= test,
                              index = "Id",
                              variable_types = {"MSSubClass": ft.variable_types.Categorical,
                                      "MSZoning": ft.variable_types.Categorical,
                                                "Street": ft.variable_types.Categorical,
                                                "Alley": ft.variable_types.Categorical,
                                                "LotShape": ft.variable_types.Categorical,
                                                "LandContour": ft.variable_types.Categorical,
#                                                "Utilities": ft.variable_types.Categorical,
                                                "LotConfig": ft.variable_types.Categorical,
                                                "LandSlope": ft.variable_types.Categorical,
                                                "Neighborhood": ft.variable_types.Categorical,
                                                "Condition1": ft.variable_types.Categorical,
                                                "Condition2": ft.variable_types.Categorical,
                                                "BldgType": ft.variable_types.Categorical,
                                                "HouseStyle": ft.variable_types.Categorical,
                                                "RoofStyle": ft.variable_types.Categorical,
                                                "RoofMatl": ft.variable_types.Categorical,
                                                "Exterior1st": ft.variable_types.Categorical,
                                                "Exterior2nd": ft.variable_types.Categorical,
                                                "MasVnrType": ft.variable_types.Categorical,
                                                "ExterQual": ft.variable_types.Categorical,
                                                "ExterCond": ft.variable_types.Categorical,
                                                "Foundation": ft.variable_types.Categorical,
                                                "BsmtQual": ft.variable_types.Categorical,
                                                "BsmtCond": ft.variable_types.Categorical,
                                                "BsmtExposure": ft.variable_types.Categorical,
                                                "BsmtFinType1": ft.variable_types.Categorical,
                                                "BsmtFinType2": ft.variable_types.Categorical,
                                                "Heating": ft.variable_types.Categorical,
                                                "HeatingQC": ft.variable_types.Categorical,
                                                "CentralAir": ft.variable_types.Categorical,
                                                "Electrical": ft.variable_types.Categorical,
                                                "KitchenQual": ft.variable_types.Categorical,
                                                "Functional": ft.variable_types.Categorical,
                                                "Fireplaces": ft.variable_types.Categorical,
                                                "FireplaceQu": ft.variable_types.Categorical,
                                                "GarageType": ft.variable_types.Categorical,
                                                "GarageFinish": ft.variable_types.Categorical,
                                                "GarageQual": ft.variable_types.Categorical,
                                                "GarageCond": ft.variable_types.Categorical,
                                                "PavedDrive": ft.variable_types.Categorical,
                                                "PoolQC": ft.variable_types.Categorical,
                                                "Fence": ft.variable_types.Categorical,
                                                "MiscFeature": ft.variable_types.Categorical,
                                                "MoSold": ft.variable_types.Categorical,
                                                "YrSold": ft.variable_types.Categorical,
                                                "SaleType": ft.variable_types.Categorical,
                                                "SaleCondition": ft.variable_types.Categorical})

#The training Set
feature_matrix, feature_defs = ft.dfs(entityset=es,
                                      target_entity = "HousingSet")
fm_encoded, features_encoded = ft.encode_features(feature_matrix,
                                                  feature_defs)
#Need to normalize the housing prices
house_prices = np.log1p(house_prices)

#Lets see labels for each
X, y = fm_encoded, house_prices

###Now testing set
feature_matrix_test, feature_defs_test = ft.dfs(entityset=es,
                                      target_entity = "HousingTest")
fm_encoded_test, features_encoded_test = ft.encode_features(feature_matrix_test,
                                                  feature_defs_test)
Actual_test = fm_encoded_test


##Fixing the alignment
X, Actual_test = X.align(Actual_test, join='left', axis=1, fill_value=0)

#Lets scale our data (Nomalize data!!!)
X = X
y = y
Actual_test = Actual_test

#CV
n_folder = 10
def rmsle_cv(model):
    kf = KFold(n_folder,shuffle = True,random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(model,X,y,scoring = 'neg_mean_squared_error',cv=kf))
    return(rmse)
score = rmsle_cv(XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1))
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


#spliting the data up
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .25)

clf_params = {
        "n_estimators":100,
             "max_depth": 100,
             "min_samples_split": 3,
             "min_samples_leaf": 3,
             "max_features": 100,
             "max_leaf_nodes": 300     
        }
clf = RandomForestRegressor(**clf_params)
scores = rmsle_cv(clf)
print("RFR score: {:.4f} ({:.4f})\n".format(scores.mean(), scores.std()))

clf.fit(X_train,y_train)
#top_features = utils.feature_importances(clf, features_encoded, n=20)
print('Variance score (Train) : %.2f' % r2_score(y_train, clf.predict(X_train)))
print("Variance score (Test): %.2f" % r2_score(y_test, clf.predict(X_test)))
mse = mean_squared_error(y_test, clf.predict(X_test))
mae = mean_absolute_error(y_test, clf.predict(X_test))
evs = explained_variance_score(y_test, clf.predict(X_test))
medae = median_absolute_error(y_test, clf.predict(X_test))
s = np.sqrt(mse)

print("Standard Error (S): %.4f" % s)
print("Mean Standard Error: %.4f" % mse)
print("Mean Absolute Error: %.4f" % mae)
print("Median Absolute Error: %.4f" % medae)
print("Explained Variance Score: %.4f" % evs)

### Using XGB
clf2 = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
clf2.fit(X_train, y_train, verbose = False)
print('Variance score (Train) : %.2f' % r2_score(y_train, clf2.predict(X_train)))
print("Variance score (Test): %.2f" % r2_score(y_test, clf2.predict(X_test)))
mse = mean_squared_error(y_test, clf2.predict(X_test))
mae = mean_absolute_error(y_test, clf2.predict(X_test))
evs = explained_variance_score(y_test, clf2.predict(X_test))
medae = median_absolute_error(y_test, clf2.predict(X_test))
s = np.sqrt(mse)

print("Standard Error (S): %.4f" % s)
print("Mean Standard Error: %.4f" % mse)
print("Mean Absolute Error: %.4f" % mae)
print("Median Absolute Error: %.4f" % medae)
print("Explained Variance Score: %.4f" % evs)

score2 = rmsle_cv(clf2)
print("XG Boosting Score: {:.4f} ({:.4f})".format(score2.mean(), score2.std()))
#GradientBoosting
clf3_params= {
        "n_estimators":1000,
             "learning_rate": .1,
             "max_depth": 10,
             "min_samples_split": 100,
             "min_samples_leaf": 3,
###             "subsample": [1],
             "max_features": 200,
             "max_leaf_nodes": 300      
        }
clf3 = GradientBoostingRegressor(**clf3_params)
clf3.fit(X_train, y_train)
print('Variance score (Train) : %.2f' % r2_score(y_train, clf3.predict(X_train)))
print("Variance score (Test): %.2f" % r2_score(y_test, clf3.predict(X_test)))
mse = mean_squared_error(y_test, clf3.predict(X_test))
mae = mean_absolute_error(y_test, clf3.predict(X_test))
evs = explained_variance_score(y_test, clf3.predict(X_test))
medae = median_absolute_error(y_test, clf3.predict(X_test))
s = np.sqrt(mse)

print("Standard Error (S): %.4f" % s)
print("Mean Standard Error: %.4f" % mse)
print("Mean Absolute Error: %.4f" % mae)
print("Median Absolute Error: %.4f" % medae)
print("Explained Variance Score: %.4f" % evs)

score3 = rmsle_cv(clf3)
print("Gradient Boosting Score: {:.4f} ({:.4f})".format(score3.mean(), score3.std()))


##Lasso Regression
model = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=1))
model.fit(X_train, y_train)
print('Variance score (Train) : %.2f' % r2_score(y_train, model.predict(X_train)))
print("Variance score (Test): %.2f" % r2_score(y_test, model.predict(X_test)))
mse = mean_squared_error(y_test, model.predict(X_test))
mae = mean_absolute_error(y_test, model.predict(X_test))
evs = explained_variance_score(y_test, model.predict(X_test))
medae = median_absolute_error(y_test, model.predict(X_test))
s = np.sqrt(mse)

print("Standard Error (S): %.4f" % s)
print("Mean Standard Error: %.4f" % mse)
print("Mean Absolute Error: %.4f" % mae)
print("Median Absolute Error: %.4f" % medae)
print("Explained Variance Score: %.4f" % evs)

model_score = rmsle_cv(model)
print("Lasso Regression Score: {:.4f} ({:.4f})".format(model_score.mean(), model_score.std()))

##KRR
KRR = KernelRidge(alpha=0.05, kernel='polynomial', degree=2, coef0=2.5)
KRR_model = rmsle_cv(KRR)
print("KRR  Score: {:.4f} ({:.4f})".format(model_score.mean(), model_score.std()))
KRR.fit(X_train, y_train)


###Another Ridge
rdg = Ridge(alpha = .999, solver = 'lsqr')
rdg.fit(X_train, y_train)

rdg_score = rmsle_cv(rdg)
print("RDG score: {:.4f} ({:.4f})".format(rdg_score.mean(), rdg_score.std()))


#This will let us look at what values are the most important
def feature_importances(model, features, n=10):
    importances = model.feature_importances_
    zipped = sorted(zip(features, importances), key=lambda x: -x[1])
    for i, f in enumerate(zipped[:n]):
        print("%d: Feature: %s, %.3f" % (i+1, f[0].get_name(), f[1]))

    return [f[0] for f in zipped[:n]]

top_features = feature_importances(clf, features_encoded, n=10)




#
##Lets optomize the paramaters
#parameters= {
#        "n_estimators":[1000],
#             "learning_rate": [.1],
#             "max_depth": [10],
#             "min_samples_split": [100],
#             "min_samples_leaf": [3, 10, 50, 100, 200, 300, 337],
#####             "subsample": [1],
##             "max_features": [200],
##             "max_leaf_nodes": [300]     
#        }

#parameters= {
#        "n_estimators":[2000],
#             "max_depth": [2, 10, 50, 100, 200, 300, 337],
##             "min_samples_split": 3,
##             "min_samples_leaf": 3,
##             "max_features": 100,
##             "max_leaf_nodes": 300     
#        }
#
#clf_grid = RandomForestRegressor()
#grid_search = GridSearchCV(clf_grid,
#                        scoring = 'neg_mean_squared_error',
#                        param_grid = parameters,
#                        cv =5)
#
#grid_search = grid_search.fit(X,y)
#est = grid_search.best_estimator_
#print(est)
#
#### Lets do a stacking of the values
#
###Now lets predict
#np.expm1 to get rid of the log normalized data
predicted_prices = np.expm1(clf2.predict(Actual_test))
predicted_prices2 = np.expm1(model.predict(Actual_test))
predicted_prices3 = np.expm1(KRR.predict(Actual_test))
Average = (predicted_prices2)
my_submission = pd.DataFrame({"Id": test.Id, "SalePrice": Average})
my_submission.to_csv('submission.csv', index = False)