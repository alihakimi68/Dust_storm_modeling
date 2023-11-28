# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 17:23:39 2023

Code to train the xgboost model on yearly data

@author: Gosia
######################################################
Updated on Nov 13 2023

Update to create windows around training and test point
for more improvement in the model

python 3.11
geopandas 0.14.1
rasterio 1.3.9
matplotlib 3.8.1
scikit-learn 1.3.2
numpy 1.26.2
pickles 0.1.1
pandas 2.1.3
xgboost 2.0.2
numpy 1.26.2

@author: hakimi.ali68@gmail.com

"""

import os
import sys
import rasterio
from rasterio.windows import Window
import geopandas as gpd
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import math as m
import pickle as pk
from scipy.spatial.distance import cdist


###################################################################
# #### Data Path ##################################################
###################################################################

os.chdir("D:/University/DustStorming/ToAli/DustStormModeling/For training/")

###################################################################
# #### Default parameters #########################################
###################################################################

CreateDataSet = True  # True for creating a dataset from For training folder
window_size = 9  # 0,3,5,7,9, ... for picking window size to search neighbor pixels of dust source
FindBestParam = False  # True for finding the best hyperparameters
year_list = list(range(2001, 2021))  # temporal duration to study 2021 is not included
CalculateSeasons = True  # divide data in to 4 periods :
# First Period is Dry from 2000:2004
# Second Period is Wet from 2005:2007
# Third Period is Dry from 2008:2012
# Fourth Period is Wet from 2012:2020

# Predict_Year = 2020 # for the predicting the whole dataset set Predict_Year = year_list

numerical = {'Mean': False,
             'WMean': True,
             'Variance': False,
             'Covariance': False,
             'Median': True}

categorical = {'Entropy': True,
               'Mode': False}

EmptyDf = pd.DataFrame(columns=['Soil evaporation', 'Lakes', 'landcover', 'Precipitation', 'Soil moisture',
                                'NDVI', 'Elevation', 'soil_type', 'Aspect', 'Curvature', 'Plan curvature',
                                'Profile curvature', 'Distance to river', 'Slope', 'dust_storm'])


Datatype = {'Soil evaporation': 'numerical', 'Lakes': 'categorical', 'landcover': 'categorical',
            'Precipitation': 'numerical', 'Soil moisture': 'numerical', 'NDVI': 'numerical',
            'Elevation': 'numerical', 'soil_type': 'categorical', 'Aspect': 'numerical',
            'Curvature': 'numerical', 'Plan curvature': 'numerical', 'Profile curvature': 'numerical',
            'Distance to river': 'numerical', 'Slope': 'numerical', 'dust_storm': 'Label'}

###################################################################
# #### Create Data set ############################################
###################################################################

if window_size == 0:
    statisticalParams = ''
else:
    true_labels_num = [label[:3] for label, value in numerical.items() if value]
    true_labels_cat = [label[:3] for label, value in categorical.items() if value]
    statisticalParams = '_'.join(true_labels_num) + '_' + '_'.join(true_labels_cat)

def crete_data_frame(EmptyDf, numerical, categorical, Datatype):

    # Create a new empty DataFrame with the desired column names
    new_columns = []
    for col in EmptyDf.columns:
        new_columns.append(col)
        if Datatype[col] == 'numerical':
            for stat, include in numerical.items():
                if include:
                    new_columns.append(f'{col} {stat.lower()}')
        elif Datatype[col] == 'categorical':
            for stat, include in categorical.items():
                if include:
                    new_columns.append(f'{col} {stat.lower()}')

    df = pd.DataFrame(columns=new_columns)

    return df

def calculate_entropy(data):
    # Convert the list to a numpy array
    data_array = np.array(data)
    # Calculate the frequency of each unique element in the array
    unique_elements, counts = np.unique(data_array, return_counts=True)
    # Calculate probabilities
    probabilities = counts / len(data)
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def createDatasetFunc(year_list,window_size,dustsourcespickle):
    if window_size == 0:
        df = EmptyDf
        dfs = []

        for year in year_list:
            print(year)

            # Load shapefile containing dust storm sources for a given year
            # shapefile_path = r"D:/ZPX/Research/DustStorm/Thesis files/Input data/For training/dust_sources/dust_training_" + str(year)+ ".shp"
            shapefile_path = r"D:/University/DustStorming/ToAli/DustStormModeling/For training/dust_sources/dust_training_" + str(
                        year) + ".shp"
            dust_storms = gpd.read_file(shapefile_path)

            # Load the rasters for the same year
            raster_paths = ["soil_evaporation/soil_evap_"+str(year)+".tif", "lakes/lakes_"+str(year),"land_use/landuse_"+str(year), "precipitation/prec_"+str(year), "soil_moisture/sm_"+str(year),"ndvi/ndvi_"+str(year), "elevation", "soil", "aspect", "curvature", "plan_c", "profile_c", "distance_riv", "slope"]

            # Extract raster values for each dust storm source/non source
            for i, row in dust_storms.iterrows():
                point = (row.geometry.x, row.geometry.y)
                values = []
                for raster_path in raster_paths:
                    # if raster_path == "soil_moisture/sm_2003": # the soil moisture data in 2003 is missing
                    #     continue
                    with rasterio.open(raster_path) as raster:
                        value = next(raster.sample([point]))[0]
                        values.append(value)
                values.append(row['ID'])
                temp_df = pd.DataFrame([values], columns=df.columns)
                dfs.append(temp_df)

                # Check if the list of DataFrames is not empty before concatenating
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            # print(df.dtypes)

    else:
        dfs = []

        df = crete_data_frame(EmptyDf, numerical, categorical, Datatype)
        # Find the dust source for every year
        for year in year_list:
        #for year in range(2001):

            print(year)
            shapefile_path = r"D:/University/DustStorming/ToAli/DustStormModeling/For training/dust_sources/dust_training_" \
                         + str(year) + ".shp"
            dust_storms = gpd.read_file(shapefile_path)

            # Load the rasters for the same year
            raster_paths = ["soil_evaporation/soil_evap_"+str(year)+".tif",
                            "lakes/lakes_"+str(year),"land_use/landuse_"+str(year),
                            "precipitation/prec_"+str(year), "soil_moisture/sm_"+str(year),
                            "ndvi/ndvi_"+str(year),
                            "elevation","soil","aspect","curvature",
                            "plan_c","profile_c","distance_riv","slope"]

            # Extract raster values for each dust storm source/non source
            for i, row in dust_storms.iterrows():
                point = (row.geometry.x, row.geometry.y)
                values = []
                flattenWindow = []

                for raster_path in raster_paths:

                    with rasterio.open(raster_path) as raster:
                        value = next(raster.sample([point]))[0]
                        values.append(value)

                        # Get the row and column indices for the point
                        row_idx, col_idx = raster.index(point[0], point[1])

                        # Define window centered at the point
                        window = Window(col_off=col_idx - (window_size//2),
                                        row_off=row_idx - (window_size//2),
                                        width=window_size,
                                        height=window_size)

                        # Read the data within the window
                        window_data = raster.read(1, window=window)

                        # Initialize an empty list to store pixel coordinates
                        all_pixel_coordinates = []

                        # Iterate over all column indices in the window
                        for col_win in range(window.col_off, window.col_off + window.width):
                            # Iterate over all row indices in the window
                            for row_win in range(window.row_off, window.row_off + window.height):
                                # Append the current pixel coordinate to the list
                                if np.abs(window_data[row_win - window.row_off, col_win - window.col_off]) < 3.4028235e+30:
                                    # print(f'row win = {row_win}')
                                    # print(f'window.row_off = {window.width}')
                                    # print(f'col_winn = {col_win}')
                                    # print(f'window.col_off = {window.height}')
                                    all_pixel_coordinates.append((col_win, row_win))

                        # Convert the list to a NumPy array if needed
                        all_pixel_coordinates = np.array(all_pixel_coordinates)
                        if all_pixel_coordinates.size == 0:
                            distances = np.zeros((1, window_size**2))
                        else:
                            distances = cdist([(col_idx, row_idx)], all_pixel_coordinates)

                        if np.any(np.isclose(window_data, -3.40282e+38)):  # check for any no data
                            window_masked = np.ma.masked_values(window_data, -3.40282e+38).compressed()
                        else:
                            window_masked = window_data
                        window_values = window_masked.flatten()
                        if window_values.size == 0:
                            # window_values = np.zeros((1, 9))
                            #window_values = np.zeros((1, 25))
                            window_values = np.zeros((1, window_size**2))

                        if raster.name in (raster_paths[1],raster_paths[2],raster_paths[7]):  # Check for categorical data

                            if categorical['Entropy']:
                                window_entropy = calculate_entropy(window_values)
                                values.append(window_entropy)

                            if categorical['Mode']:
                                non_negative_values = [val for val in window_values if val >= 0]
                                if non_negative_values:
                                    window_Mode = np.argmax(np.bincount(np.array(non_negative_values).astype(int)))
                                    values.append(window_Mode)
                                else:
                                    window_Mode = np.nan
                                    values.append(window_Mode)

                        else:
                            if numerical['Mean']:
                                window_average = round(window_values.mean(),8)
                                values.append(window_average)

                            if numerical['WMean']:

                                zero_indices = np.where(distances == 0)[1]
                                if len(zero_indices) > 0:
                                    # Remove the zero values from the distances array
                                    distances = distances[distances != 0]
                                    window_values_avr = np.delete(window_values, zero_indices)
                                else:
                                    window_values_avr = np.array(window_values)
                                if np.all(distances == 0):
                                    weighted_average = 0
                                    values.append(weighted_average)
                                else:
                                    #     window_values = np.array(window_values)
                                    weights = (1 / distances**2).flatten()

                                    weighted_average = np.average(window_values_avr, weights=weights)
                                    # window_average = round(window_values.mean(),8)
                                    values.append(weighted_average)

                            if numerical['Variance']:
                                window_variance = round(window_values.var(),8)
                                values.append(window_variance)

                            if numerical['Covariance']:
                                if len(window_values) == window_size**2:
                                    window_matrix = window_values.reshape((window_size, window_size))

                                    # Calculate the covariance matrix
                                    covariance_matrix = np.cov(window_matrix)

                                    if np.any(np.isnan(covariance_matrix)):
                                        covariance_value = 0
                                    else:
                                        covariance_value = covariance_matrix[window_size // 2, window_size // 2]
                                else:
                                    covariance_value = 0
                                values.append(covariance_value)

                            if numerical['Median']:
                                window_median = np.median(window_values)
                                window_median = window_median.astype(np.float32)
                                values.append(window_median)

                            # # Calculate the correlation coefficient matrix
                            # correlation_matrix = np.corrcoef(window_matrix)
                            #
                            # if np.any(np.isnan(correlation_matrix)):
                            #     correlation_coefficient = 0
                            # else:
                            #     # Check for very small standard deviations to avoid division by almost zero
                            #     std_dev_threshold = 1e-10
                            #     if np.any(np.abs(np.std(window_matrix, axis=0)) < std_dev_threshold) or np.any(
                            #             np.abs(np.std(window_matrix, axis=1)) < std_dev_threshold):
                            #         correlation_coefficient = 0
                            #     else:
                            #         correlation_coefficient = np.mean(correlation_matrix)
                            # values.append(correlation_coefficient)


                            # print(f'{raster.name}, window_average={window_average}, window_variance={window_variance}')

                values.append(row['ID'])
                # values.append(year)

                temp_df = pd.DataFrame([values], columns=df.columns)
                dfs.append(temp_df)

        # Check if the list of DataFrames is not empty before concatenating
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        # print(df.dtypes)

    pk.dump(df, open(f'{dustsourcespickle}.pickle','wb'))


if CreateDataSet:
    print(f'CreateDataSet is set to {CreateDataSet}')



    if CalculateSeasons:
        print(f'CalculateSeasons is set to {CalculateSeasons}')
        periods = [year_list[0:4],year_list[4:7],year_list[7:12],year_list[12:20]] # For Four Periods 2 Dry and 2Wet
        # periods = [year_list[0:4] + year_list[7:12], year_list[4:7] + year_list[12:20]] # FOr 2 Periods 1 Dry and 1 Wet
        for period in periods:
            year_list_period = period
            print(f'Creating Data set for {year_list_period}')
            # First Period is Dry from 2000:2004
            # Second Period is Wet from 2005:2007
            # Third Period is Dry from 2008:2012
            # Fourth Period is Wet from 2012:2020
            PeriodName = len(year_list_period)

            dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}'

            createDatasetFunc(year_list_period, window_size,dustsourcespickle)
    else:
        print(f'CalculateSeasons is set to {CreateDataSet}')
        PeriodName = len(year_list)
        print(f'Creating Data set for {year_list}')
        dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}'
        createDatasetFunc(year_list, window_size,dustsourcespickle)

###################################################################
# #### Load Data set ##############################################
###################################################################


def loadDatSet(df,window_size):
    print(f'windows size: {window_size}')

    # Replace the no data value with nan in the dataframe
    df.replace(-3.4028234663852886e+38, np.nan, inplace=True)

    # Count the occurrences of each value in the dust_storm column
    dust_storm_counts = df['dust_storm'].value_counts()

    # Define the mapping dictionary for each dataset
    landuse_mapping = {0: 'Water', 1: 'Natural vegetation', 3: 'Cropland', 4: 'Urban', 6: 'Bare Soil'}
    soil_type_mapping = {0: 'Silt', 1: 'Clay', 2: 'Silt Clay', 3: 'Sand Clay', 4: 'Clay Loam', 5: 'Silt Clay Loam',
                         6: 'Sand Clay Loam', 7: 'Loam', 8: 'Silt Loam', 9: 'Sand Loam', 11: 'Loam Sand', 12: 'Sand'}

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
    # else:
    #     print("Column 255.0 not found!")

    if -1 in df.columns:
        df.drop(-1, axis=1, inplace=True)
    # else:
    #     print("Column -1 not found!")

    X = df.drop(['dust_storm'], axis=1)
    y = df['dust_storm']

    df.to_csv('training.csv', index=False)
    # Split data to 70% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)
    return X_train, X_test, y_train, y_test, X, y

def fitTheModelXGboost(X_train, X_test, y_train, y_test,X, y):
    ###################################################################
    # #### Fit the model and result ###################################
    ###################################################################
    # Set hyperparameters
    params = {}
    params['objective'] = 'binary:logistic'
    params['num_class'] = 1
    params['eval_metric'] = 'auc'
    if CalculateSeasons:
        params['learning_rate'] = 0.02
        # window size = 5: max_depth=7, window size = 7: max_depth=10
        if window_size == 3:
            params['max_depth'] = 5
        elif window_size == 5:
            params['max_depth'] = 8
        elif window_size == 7:
            params['max_depth'] = 11
        elif window_size == 9:
            params['max_depth'] = 11
        elif window_size == 11:
            params['max_depth'] = 11
        else:
            params['max_depth'] = 5
        params['min_child_weight'] = 2
        params['reg_alpha'] = 0.6
        params['reg_lambda'] = 0.7
        params['subsample'] = 0.7
        params['gamma'] = 0.2
        params['num_parallel_tree'] = 2

    else:
        params['learning_rate'] = 0.01
        # window size = 5: max_depth=7, window size = 7: max_depth=10
        if window_size == 3:
            params['max_depth'] = 5
        elif window_size == 5:
            params['max_depth'] = 8
        elif window_size == 7:
            params['max_depth'] = 11
        elif window_size == 9:
            params['max_depth'] = 11
        elif window_size == 11:
            params['max_depth'] = 11
        else:
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
    # y_pred = xgb_model.predict(X_val) # validation set
    y_pred = xgb_model.predict(X_test)  # final test set



    # Evaluate the metrics to check accuracy, precision, recall, f1-score, confusion matrix and AUC
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # Print the metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision: {:.2f}%".format(precision * 100))
    print("Recall: {:.2f}%".format(recall * 100))
    print("F1-score: {:.2f}%".format(f1 * 100))
    print('Confusion matrix:\n True negative: %s \
              \n False positive: %s \n False negative: %s \n True positive: %s'
          % (conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]))
    print('AUC: {:.2f}%'.format(auc * 100))

    # Cross validation
    scores = cross_val_score(xgb_model, X, y, cv=5)
    print('The cross validation accuracies of the model are', scores)
    print('The cross validation accuracy of the model is', np.mean(scores))

    # Save the current standard output
    original_stdout = sys.stdout

    # Specify the file path where you want to save the results
    file_path = f'{dustsourcespickle}_Results.txt'

    # Open the file in write mode
    with open(file_path, 'w') as file:
        # Redirect standard output to the file
        sys.stdout = file

        # Print the column names
        print(
            "Accuracy\tPrecision\tRecall\tF1-score\tTrue negative\tFalse positive\tFalse negative\tTrue positive\tAUC")

        # Print the values in the second line
        print("{:.2f}%\t{:.2f}%\t{:.2f}%\t{:.2f}%\t{}\t{}\t{}\t{}\t{:.2f}%".format(
            accuracy * 100, precision * 100, recall * 100, f1 * 100,
            conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1],
            auc * 100
        ))

        # Cross validation
        scores = cross_val_score(xgb_model, X, y, cv=5)
        print('\nThe cross validation accuracies of the model are', scores)
        print('The cross validation accuracy of the model is', np.mean(scores))
        print('\nThe windows size is', window_size)
        print('\nThe results are for the years:', year_list)
    # Reset the standard output to the original
    sys.stdout = original_stdout

    # Print a message indicating that the results have been saved
    print(f'Results have been saved to {file_path}')

    fig, ax = plt.subplots(figsize=(20, 16))  # Set the figure size (adjust the values as needed)
    xgb.plot_importance(xgb_model, ax=ax)
    plt.savefig(f'{dustsourcespickle}_Results.png', bbox_inches='tight')  # Save the plot as an image file
    plt.show()



if CalculateSeasons:
    print(f'CalculateSeasons is set to {CreateDataSet}')
    periods = [year_list[0:4], year_list[4:7], year_list[7:12], year_list[12:20]] # For Four Periods 2 Dry and 2Wet
    # periods = [year_list[0:4] + year_list[7:12], year_list[4:7] + year_list[12:20]]  # FOr 2 Periods 1 Dry and 1 Wet
    for period in periods:
        year_list_period = period
        print(f'Calculating the period:{year_list_period}')
        PeriodName = len(year_list_period)
        dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}'
        df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))
        X_train, X_test, y_train, y_test, X, y = loadDatSet(df,window_size)
        fitTheModelXGboost(X_train, X_test, y_train, y_test,X, y)
else:
    print(f'CalculateSeasons is set to {CreateDataSet}')
    PeriodName = len(year_list)
    print(f'Calculating the period:{year_list}')
    dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}'
    df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))
    X_train, X_test, y_train, y_test, X, y = loadDatSet(df, window_size)
    fitTheModelXGboost(X_train, X_test, y_train, y_test,X, y)

###################################################################
# #### HYPERPARAMETER TUNING ######################################
###################################################################

if FindBestParam:

    # RandomizedSearchCV

    #Define the parameter space to search over
    param_grid = {
       'max_depth': np.arange(8, 12),
       'min_child_weight': np.arange(1, 3),
       'subsample': np.arange(0.5, 0.9, 0.1),
       'gamma': np.arange(0, 0.6, 0.1),
       'reg_alpha': np.arange(0.4, 1.0, 0.1),
       'reg_lambda': np.arange(0.4, 1.0, 0.1),
       'learning_rate': np.arange(0.01, 0.2)
    }

    # Create an instance of XGBoost classifier
    tuning_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', n_jobs=-1)

    # gs = GridSearchCV(tuning_model, param_grid=param_grid, cv=4, n_jobs=2 )
    #
    # gs.fit(X_train,y_train)
    #
    # print(gs.best_params_)

    # Create an instance of RandomizedSearchCV
    rs = RandomizedSearchCV(tuning_model, param_distributions=param_grid, n_iter=400000, cv=4, verbose=0, random_state=42, n_jobs=-1)

    # Fit the model to the data
    rs.fit(X_train, y_train)

    # Print the best hyperparameters
    print('Best hyperparameters:', rs.best_params_)


