# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:23:39 2023

Code to train the xgboost model on yearly data

@author: Gosia
"""
import os
import rasterio
import geopandas as gpd
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif, RFE
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import math as m
import pickle as pk

os.chdir("D:/University/DustStorming/ToAli/DustStormModeling/For training/")

#
# # Create an empty dataframe to store extracted values. This part is to generate training data including dust sources and non dust sources
# df = pd.DataFrame(columns=['Soil evaporation', 'Lakes', 'landcover', 'Precipitation', 'Soil moisture', 'NDVI', 'Elevation', 'soil_type', 'Aspect', 'Curvature', 'Plan curvature', 'Profile curvature', 'Distance to river', 'Slope', 'dust_storm'])
#
# for year in range(2001, 2021):
#     print(year)
#     if year == 2003:
#         continue
#     # Load shapefile containing dust storm sources for a given year
#     # shapefile_path = r"D:/ZPX/Research/DustStorm/Thesis files/Input data/For training/dust_sources/dust_training_" + str(year)+ ".shp"
#     shapefile_path = r"D:/University/DustStorming/ToAli/DustStormModeling/For training/dust_sources/dust_training_" + str(
#                 year) + ".shp"
#     dust_storms = gpd.read_file(shapefile_path)
#
#     # Load the rasters for the same year
#     raster_paths = ["soil_evaporation/soil_evap_"+str(year)+".tif", "lakes/lakes_"+str(year),"land_use/landuse_"+str(year), "precipitation/prec_"+str(year), "soil_moisture/sm_"+str(year),"ndvi/ndvi_"+str(year), "elevation", "soil", "aspect", "curvature", "plan_c", "profile_c", "distance_riv", "slope"]
#
#     # Extract raster values for each dust storm source/non source
#     for i, row in dust_storms.iterrows():
#         point = (row.geometry.x, row.geometry.y)
#         values = []
#         for raster_path in raster_paths:
#             # if raster_path == "soil_moisture/sm_2003": # the soil moisture data in 2003 is missing
#             #     continue
#             with rasterio.open(raster_path) as raster:
#                 value = next(raster.sample([point]))[0]
#                 values.append(value)
#         values.append(row['ID'])
#         temp_df = pd.DataFrame([values], columns=df.columns)
#         df = pd.concat([df, temp_df], ignore_index=True)
#         # df = df.append(pd.Series(values, index=df.columns), ignore_index=True)
#
# pk.dump(df, open('_df_dustsources.pickle','wb'))
#
df = pk.load(open('_df_dustsources.pickle','rb'))
# Replace the no data value with nan in the dataframe
df.replace(-3.4028234663852886e+38, np.nan, inplace=True)

# Count the occurrences of each value in the dust_storm column
dust_storm_counts = df['dust_storm'].value_counts()


# Define the mapping dictionary for each dataset
landuse_mapping = {0: 'Water', 1: 'Natural vegetation', 3: 'Cropland', 4: 'Urban', 6: 'Bare Soil'}
soil_type_mapping = {0: 'Silt', 1: 'Clay', 2: 'Silt Clay', 3: 'Sand Clay', 4: 'Clay Loam', 5: 'Silt Clay Loam', 6: 'Sand Clay Loam', 7: 'Loam', 8: 'Silt Loam', 9: 'Sand Loam', 11: 'Loam Sand', 12: 'Sand'}

# Replace the numerical values with their corresponding category names for dataset1
df.replace({'landcover': landuse_mapping, 'soil_type': soil_type_mapping}, inplace=True)

# Create dummy variables for each categorical column in the dataframe
dummy_landcover = pd.get_dummies(df['landcover'])
dummy_soil_type = pd.get_dummies(df['soil_type'])

# Add missing columns to dataframe with a value of 0
for col in soil_type_mapping.values():
    if col not in dummy_soil_type.columns:
        dummy_soil_type[col] = 0

# concatonate both dummy dataframes
dummy_df = pd.concat([dummy_landcover, dummy_soil_type], axis=1)

# concatenate dummy variables to original dataframe
df = pd.concat([df, dummy_df], axis=1)

# drop original categorical columns
df = df.drop(columns=['landcover', 'soil_type'])

# drop not important columns
df = df.drop(columns=['Water', 'Clay', 'Silt', 'Silt Loam', 'Silt Clay', 'Sand Clay', 'Silt Clay Loam', 'Urban'])


if 255.0 in df.columns:
    df.drop(255.0, axis=1, inplace=True)

if -1 in df.columns:
    df.drop(-1, axis=1, inplace=True)

df['Lakes'] = df['Lakes'].astype('float32')
df['Elevation'] = df['Elevation'].astype('float32')
df['dust_storm'] = df['dust_storm'].astype('float32')


X = df.drop(['dust_storm'], axis=1)
y = df['dust_storm']
df.to_csv('training.csv', index=False)
# Split data to 70% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

##### HYPERPARAMETER TUNING ##############################################
# RandomizedSearchCV

# Define the parameter space to search over
#param_grid = {
#    'max_depth': np.arange(3, 5),
#    'min_child_weight': np.arange(1, 3),
#    'subsample': np.arange(0.5, 0.9, 0.1),
#    'gamma': np.arange(0, 0.6, 0.1),
#    'reg_alpha': np.arange(0.4, 1.0, 0.1),
#    'reg_lambda': np.arange(0.4, 1.0, 0.1),
#    'learning_rate': np.arange(0.01, 0.2)
#}

#Create an instance of XGBoost classifier
#tuning_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', n_jobs=-1)

#gs = GridSearchCV(tuning_model, param_grid=param_grid, cv=4, n_jobs=2 )

#gs.fit(X_train,y_train)

#print(gs.best_params_)

# Create an instance of RandomizedSearchCV
#rs = RandomizedSearchCV(tuning_model, param_distributions=param_grid, n_iter=400000, cv=4, verbose=2, random_state=42, n_jobs=-1)

# Fit the model to the data
#rs.fit(X_train, y_train)

# Print the best hyperparameters
#print('Best hyperparameters:', rs.best_params_)


##########################################################################

# Set hyperparameters
params = {}
params['objective'] = 'binary:logistic'
params['num_class'] = 1
params['eval_metric'] = 'auc'
params['learning_rate'] = 0.02
params['max_depth'] = 5
params['min_child_weight'] = 2
params['reg_alpha'] = 0.6
params['reg_lambda'] = 0.7
params['subsample'] = 0.7
params['gamma'] = 0.2
params['num_parallel_tree'] = 2

# Create an XGBoost classifier with the hyperparameters dictionary
xgb_model = xgb.XGBClassifier(**params)



# Fit the XGBoost model to the training data
xgb_model.fit(X_train, y_train)

# Predict class labels
#y_pred = xgb_model.predict(X_val) # validation set
y_pred = xgb_model.predict(X_test) # final test set


# Evaluate the metrics to check accuracy, precision, recall, f1-score, confusion matrix and AUC
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)


# Print the metrics
print("Accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))
print("F1-score: {:.2f}%".format(f1*100))
print('Confusion matrix:\n True negative: %s \
      \n False positive: %s \n False negative: %s \n True positive: %s'
       % (conf_matrix[0,0], conf_matrix[0,1], conf_matrix[1,0], conf_matrix[1,1]))
print('AUC: {:.2f}%'.format(auc*100))


# Cross validation
scores = cross_val_score(xgb_model, X, y, cv=5)
print('The cross validation accuracies of the model are', scores)
print('The cross validation accuracy of the model is', np.mean(scores))


# Plot feature importance
xgb.plot_importance(xgb_model)
plt.show()








