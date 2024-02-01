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
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif, RFE
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score,make_scorer)
from scikeras.wrappers import KerasClassifier
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

CreateDataSet = False  # True for creating a dataset from For training folder
window_size = 7  # 0,3,5,7,9, ... for picking window size to search neighbor pixels of dust source
FindBestParam = False  # True for finding the best hyperparameters
year_list = list(range(2001, 2021))  # temporal duration to study 2021 is not included
CalculateSeasons = False  # divide data in to 4 periods :
# First Period is Dry from 2000:2004
# Second Period is Wet from 2005:2007
# Third Period is Dry from 2008:2012
# Fourth Period is Wet from 2012:2020

# Predict_Year = 2020 # for the predicting the whole dataset set Predict_Year = year_list

numerical = {'Mean': False,
             'WMean': True,
             'Variance': False,
             'Covariance': False,
             'Median': False}

categorical = {'Entropy': False,
               'Mode': False}

EmptyDf = pd.DataFrame(columns=['Soil_evaporation', 'Lakes', 'landcover', 'Precipitation', 'Soil_moisture',
                                'NDVI', 'Elevation', 'soil_type', 'Aspect', 'Curvature', 'Plan_curvature',
                                'Profile_curvature', 'Distance_to_river', 'Slope', 'dust_storm',
                                'Year','X','Y'])


Datatype = {'Soil_evaporation': 'numerical', 'Lakes': 'categorical', 'landcover': 'categorical',
            'Precipitation': 'numerical', 'Soil_moisture': 'numerical', 'NDVI': 'numerical',
            'Elevation': 'numerical', 'soil_type': 'categorical', 'Aspect': 'numerical',
            'Curvature': 'numerical', 'Plan_curvature': 'numerical', 'Profile_curvature': 'numerical',
            'Distance_to_river': 'numerical', 'Slope': 'numerical', 'dust_storm': 'Label',
            'Year':'Year','X':'X','Y':'Y'}

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
                values.append(year)
                values.append(point[0])
                values.append(point[1])
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

                                # if len(window_values) == window_size**2:
                                #     window_matrix = window_values.reshape((window_size, window_size))
                                #     # Calculate the covariance matrix with weights (distances)
                                #     print(window_matrix)
                                #
                                #     weights = 1 / distances.flatten()**2
                                #     weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                                #     weights = weights.reshape(window_values.shape)
                                #
                                #     # Print the sizes for debugging
                                #     print("Size of window_matrix:", window_matrix.shape)
                                #     print("Size of weights:", weights.shape)
                                #     covariance_matrix = np.cov(window_matrix, aweights=weights)
                                #
                                #     # if np.any(np.isnan(covariance_matrix)):
                                #     #     covariance_value = 0
                                #     # else:
                                #     covariance_value = covariance_matrix[window_size // 2, window_size // 2]
                                # else:
                                #     covariance_value = 0



                            if numerical['Median']:
                                window_median = np.median(window_values)
                                window_median = window_median.astype(np.float32)
                                values.append(window_median)

                            # print(f'{raster.name}, window_average={window_average}, window_variance={window_variance}')

                values.append(row['ID'])
                values.append(year)
                values.append(point[0])
                values.append(point[1])

                temp_df = pd.DataFrame([values], columns=df.columns)
                dfs.append(temp_df)

        # Check if the list of DataFrames is not empty before concatenating
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        # print(df.dtypes)

    # Replace the no data value with nan in the dataframe
    df.replace(-3.4028234663852886e+38, np.nan, inplace=True)

    # Count the occurrences of each value in the dust_storm column
    dust_storm_counts = df['dust_storm'].value_counts()

    # Define the mapping dictionary for each dataset
    landuse_mapping = {0: 'Water', 1: 'Natural_vegetation', 3: 'Cropland', 4: 'Urban', 6: 'Bare_Soil'}
    soil_type_mapping = {0: 'Silt', 1: 'Clay', 2: 'Silt_Clay', 3: 'Sand_Clay', 4: 'Clay_Loam', 5: 'Silt_Clay_Loam',
                         6: 'Sand_Clay_Loam', 7: 'Loam', 8: 'Silt_Loam', 9: 'Sand_Loam', 11: 'Loam_Sand',
                         12: 'Sand'}

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

    # drop not important columns
    df = df.drop(
        columns=['Water', 'Clay', 'Silt', 'Silt_Loam', 'Silt_Clay', 'Sand_Clay', 'Silt_Clay_Loam', 'Urban'])

    if 255.0 in df.columns:
        df.drop(255.0, axis=1, inplace=True)
    # else:
    #     print("Column 255.0 not found!")

    if (255 in df.values):
        # Drop rows containing -1 in any attribute
        df = df[df.apply(lambda row: 255 not in row.values, axis=1)]

    if -1 in df.columns:
        df.drop(-1, axis=1, inplace=True)

    if (-1 in df.values):
        # Drop rows containing -1 in any attribute
        df = df[df.apply(lambda row: -1 not in row.values, axis=1)]

    if df.isna().any().any():
        # Drop rows containing NaN values in any attribute
        df = df.dropna()


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
        print(f'CalculateSeasons is set to {CalculateSeasons}')
        PeriodName = len(year_list)
        print(f'Creating Data set for {year_list}')
        dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}'
        createDatasetFunc(year_list, window_size,dustsourcespickle)

###################################################################
# #### Load Data set ##############################################
###################################################################


def loadDatSet(df,window_size):
    print(f'windows size: {window_size}')

    # drop original categorical columns
    df = df.drop(columns=['X', 'Y','Year', 'landcover', 'soil_type'])

    df.to_csv(f'{dustsourcespickle}.csv', index=False)

    columns_to_scale = ["Soil_evaporation", "Lakes", "Precipitation", "Soil_moisture",
                        "NDVI", "Elevation", "Aspect", "Curvature", "Plan_curvature",
                        "Profile_curvature", "Distance_to_river", "Slope"]

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()

    # Fit and transform the specified columns
    # df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
    df[df.columns] = scaler.fit_transform(df[df.columns])

    X = df.drop(['dust_storm'], axis=1)
    # X = df.drop(['dust_storm','Lakes entropy','Loam Sand','Sand Clay Loam','Clay Loam',
    #              'Bare Soil','Cropland','Natural vegetation','Lakes mode','Precipitation median'], axis=1)

    y = df['dust_storm']
    value_counts = y.value_counts()
    # y = to_categorical(y, num_classes=2)

    print(f'number of dust sources {value_counts.get(1, 0)}')
    print(f'number of none dust sources {value_counts.get(0, 0)}')
    df.to_csv(f'{dustsourcespickle}.csv', index=False)
    # Split data to 80% training, 20% testing


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)

    return X_train, X_test, y_train, y_test, X, y

def fit_the_model_ann(X_train, X_test, y_train, y_test, X, y):
    ###################################################################
    # #### Fit the model and result ###################################
    ###################################################################
    # Create the ANN model
    input_dim = X_train.shape[1]

    ann_model = tf.keras.Sequential()
    ann_model.add(tf.keras.layers.Dense(units=128, activation='relu', input_dim=input_dim))
    ann_model.add(tf.keras.layers.Dropout(0.5))
    ann_model.add(tf.keras.layers.Dense(units=64, activation='relu'))
    ann_model.add(tf.keras.layers.Dropout(0.5))
    ann_model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    # Fit the ANN model to the training data
    history = ann_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Evaluate the ANN model on the test set
    loss, accuracy = ann_model.evaluate(X_test, y_test)
    print(f'ANN Model - Test Loss: {loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%')

    # Save the ANN model
    ann_model.save('ann_model.keras')

    y_pred_probs = ann_model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

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
    # # Wrap the Keras model in a scikit-learn compatible wrapper
    # ann_model_wrapper = KerasClassifier(build_fn=ann_model, epochs=10, batch_size=32,
    #                                     validation_data=(X_test, y_test))
    #
    # # Specify the scoring metric you want to use
    # scoring = {'accuracy': make_scorer(accuracy_score)}
    # scores = cross_val_score(ann_model_wrapper, X, y, cv=6, scoring=scoring['accuracy'])
    # print('The cross validation accuracies of the model are', scores)
    # print('The cross validation accuracy of the model is', np.mean(scores))

    # Save the current standard output
    original_stdout = sys.stdout

    # Specify the file path where you want to save the results
    file_path = f'{dustsourcespickle}_Results_ANN.txt'

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
        # scores = cross_val_score(ann_model, X, y, cv=5)
        # print('\nThe cross validation accuracies of the model are', scores)
        # print('The cross validation accuracy of the model is', np.mean(scores))
        print('\nThe windows size is', window_size)
        print('\nThe results are for the years:', year_list)
    # Reset the standard output to the original
    sys.stdout = original_stdout

    # Print a message indicating that the results have been saved
    print(f'Results have been saved to {file_path}')

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    train_acc = history.history['accuracy']

    # Plotting
    # Plot the training and validation loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    # Show the plots
    plt.tight_layout()
    plt.savefig(f'{dustsourcespickle}_Results_ANN.png', bbox_inches='tight')

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
        fit_the_model_ann(X_train, X_test, y_train, y_test, X, y)
else:
    print(f'CalculateSeasons is set to {CreateDataSet}')
    PeriodName = len(year_list)
    print(f'Calculating the period:{year_list}')

    # for the normal routine
    dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}'
    df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))

    # # for Hyper parameter tuning
    # dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod.pickle'
    # df = pk.load(open(f'{dustsourcespickle}', 'rb'))

    print(f'Loading {dustsourcespickle} as dataset')
    X_train, X_test, y_train, y_test, X, y = loadDatSet(df, window_size)
    fit_the_model_ann(X_train, X_test, y_train, y_test, X, y)


###################################################################
# #### HYPERPARAMETER TUNING ######################################
###################################################################

if FindBestParam:


    #Define the parameter space to search over
    param_grid = {
       'max_depth': np.arange(5, 15),
       'min_child_weight': np.arange(1, 3),
       'subsample': np.arange(0.2, 0.9, 0.1),
       'gamma': np.arange(0, 2, 0.1),
       'reg_alpha': np.arange(0.4, 1.0, 0.1),
       'reg_lambda': np.arange(0.4, 1.0, 0.1),
       'learning_rate': np.arange(0.01, 0.2)
    }

    # param_grid = {
    #     'max_depth': np.arange(5, 15),
    #     'min_child_weight': np.arange(1, 3),
    #     'subsample': np.arange(0.5, 0.9, 0.1),
    #     'gamma': np.arange(0, 0.8, 0.1),
    #     'reg_alpha': np.arange(0.4, 1.0, 0.1),
    #     'reg_lambda': np.arange(0.4, 1.0, 0.1),
    #     'learning_rate': np.arange(0.01, 0.2)
    # }

    # Create an instance of XGBoost classifier
    tuning_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', n_jobs=-1)
    #
    # gs = GridSearchCV(tuning_model, param_grid=param_grid, cv=4, n_jobs=2 )
    #
    # gs.fit(X_train,y_train)



    # Create an instance of RandomizedSearchCV
    rs = RandomizedSearchCV(tuning_model, param_distributions=param_grid, n_iter=100800, cv=6, verbose=0, random_state=42, n_jobs=-1)

    # Fit the model to the data
    rs.fit(X_train, y_train)
    # print('GridSearchCV Best Parameters')
    # print(gs.best_params_)
    # Print the best hyperparameters
    print('RandomizedSearchCV Best Parameters')
    print('Best hyperparameters:', rs.best_params_)


