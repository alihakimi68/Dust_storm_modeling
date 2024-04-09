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
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import math as m
import pickle as pk
from scipy.spatial.distance import cdist
from rasterio.transform import from_origin
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import OneHotEncoder


###################################################################
# #### Data Path ##################################################
###################################################################

os.chdir("/For training/")

###################################################################
# #### Default parameters #########################################
###################################################################

CreateDataSet = True  # True for creating a dataset from For training folder
window_size = 0  # 0,3,5,7,9, ... for picking window size to search neighbor pixels of dust source
year_list = list(range(2001, 2021))  # temporal duration to study 2021 is not included
CalculateSeasons = False  # divide data in to 4 periods :
CalculateDistance = False
# First Period is Dry from 2000:2004
# Second Period is Wet from 2005:2007
# Third Period is Dry from 2008:2012
# Fourth Period is Wet from 2012:2020

# Predict_Year = 2020 # for the predicting the whole dataset set Predict_Year = year_list

numerical = {'Mean': False,
             'WMean': False,
             'Variance': False,
             'Covariance': False,
             'Median': False}

categorical = {'Entropy': False,
               'Mode': False}

EmptyDf = pd.DataFrame(columns=['Soil_evaporation', 'Lakes', 'landcover', 'Precipitation', 'Soil_moisture',
                                'NDVI', 'Elevation', 'soil_type', 'Aspect', 'Curvature', 'Plan_curvature',
                                'Profile_curvature', 'Distance_to_river', 'Slope', 'dust_storm',
                                'Year','X','Y'])

EmptyDf_Dist = pd.DataFrame(columns=['Soil_evaporation', 'Lakes','Distance_to_lakes', 'landcover','Distance_to_cropland', 'Precipitation', 'Soil_moisture',
                                'NDVI','Distance_to_SparseVegetation','Distance_to_DenseVegetation', 'Elevation', 'soil_type', 'Aspect', 'Curvature', 'Plan_curvature',
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

def haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # Radius of Earth in kilometers (you can change this to miles if needed)
    R = 6371.0

    # Calculate the distance
    distance = R * c

    return distance



def createDatasetFunc(year_list,window_size,dustsourcespickle):

    # if window_size == 0 and CalculateDistance:
    #     df = EmptyDf_Dist
    #     dfs = []
    #
    #     for year in year_list:
    #         print(year)
    #
    #         # Load shapefile containing dust storm sources for a given year
    #         # shapefile_path = r"D:/ZPX/Research/DustStorm/Thesis files/Input data/For training/dust_sources/dust_training_" + str(year)+ ".shp"
    #         shapefile_path = r"D:/University/DustStorming/ToAli/DustStormModeling/For training/dust_sources/dust_training_" + str(
    #                     year) + ".shp"
    #         dust_storms = gpd.read_file(shapefile_path)
    #
    #         # Load the rasters for the same year
    #         raster_paths = ["soil_evaporation/soil_evap_"+str(year)+".tif", "lakes/lakes_"+str(year),"land_use/landuse_"+str(year), "precipitation/prec_"+str(year), "soil_moisture/sm_"+str(year),"ndvi/ndvi_"+str(year), "elevation", "soil", "aspect", "curvature", "plan_c", "profile_c", "distance_riv", "slope"]
    #
    #         # Extract raster values for each dust storm source/non source
    #         for i, row in dust_storms.iterrows():
    #             point = (row.geometry.x, row.geometry.y)
    #             lat_dust_source, lon_dust_source = point
    #             values = []
    #             for raster_path in raster_paths:
    #                 # if raster_path == "soil_moisture/sm_2003": # the soil moisture data in 2003 is missing
    #                 #     continue
    #                 with rasterio.open(raster_path) as raster:
    #                     value = next(raster.sample([point]))[0]
    #                     values.append(value)
    #
    #                     if "lakes/lakes" in raster_path:
    #                         raster_transform = raster.transform
    #                         # If the current raster is the lakes raster, find the closest value of 1
    #                         # Load the lake raster as a numpy array
    #                         lake_data = raster.read(1)
    #
    #                         # Get the coordinates of all 1 values in the lake raster
    #                         lake_coords = np.argwhere(lake_data == 1)
    #
    #                         # Calculate distances between the current point and all lake coordinates
    #                         distances_lakes = cdist([point], lake_coords)
    #
    #                         # Find the index of the minimum distance
    #                         min_distance_index = np.argmin(distances_lakes)
    #
    #                         # Get the coordinates of the closest 1 value in the lake raster
    #                         closest_lake_coords = lake_coords[min_distance_index]
    #
    #                         lon_closest_lake, lat_closest_lake = raster_transform * (
    #                         closest_lake_coords[1], closest_lake_coords[0])
    #                         distance_km = haversine(lat_dust_source, lon_dust_source, lat_closest_lake,
    #                                                 lon_closest_lake)
    #                         values.append(distance_km)
    #
    #
    #                     if "land_use/landuse" in raster_path:
    #                         raster_transform = raster.transform
    #                         # If the current raster is the lakes raster, find the closest value of 1
    #                         # Load the lake raster as a numpy array
    #                         Landuse_data = raster.read(1)
    #
    #                         # Get the coordinates of all 1 values in the lake raster
    #                         Cropland_coords = np.argwhere(Landuse_data == 3)
    #
    #                         # Calculate distances between the current point and all lake coordinates
    #                         Cropland_distances = cdist([point], Cropland_coords)
    #
    #                         # Find the index of the minimum distance
    #                         min_distance_index_Cropland = np.argmin(Cropland_distances)
    #
    #                         # Get the coordinates of the closest 1 value in the lake raster
    #                         closest_coords_Cropland = Cropland_coords[min_distance_index_Cropland]
    #
    #                         lon_closest_Cropland, lat_closest_Cropland = raster_transform * (
    #                             closest_coords_Cropland[1], closest_coords_Cropland[0])
    #                         distance_km_Cropland = haversine(lat_dust_source, lon_dust_source, lat_closest_Cropland,
    #                                                 lon_closest_Cropland)
    #                         values.append(distance_km_Cropland)
    #
    #
    #                     if "ndvi/ndvi_" in raster_path:
    #                         # If the current raster is the NDVI raster
    #                         raster_transform = raster.transform
    #                         ndvi_data = raster.read(1)
    #
    #
    #                         # distance to Sparse vegetation
    #                         # Find the indices of NDVI values within the range [0.1, 0.5]
    #                         valid_ndvi_indices = np.where((ndvi_data >= 0.1) & (ndvi_data <= 0.5))
    #
    #                         # Create a list of coordinates within the specified NDVI range
    #                         valid_ndvi_coords = list(zip(valid_ndvi_indices[0], valid_ndvi_indices[1]))
    #
    #                         # Calculate distances between the current point and all valid NDVI coordinates
    #                         distances_ndvi = cdist([point], valid_ndvi_coords)
    #
    #                         # Find the index of the minimum distance
    #                         min_distance_index_ndvi = np.argmin(distances_ndvi)
    #
    #                         # Get the coordinates of the closest NDVI value in the specified range
    #                         closest_ndvi_coords = valid_ndvi_coords[min_distance_index_ndvi]
    #
    #                         # Convert pixel coordinates to geographical coordinates
    #                         lon_closest_ndvi, lat_closest_ndvi = raster_transform * (
    #                         closest_ndvi_coords[1], closest_ndvi_coords[0])
    #
    #                         # Calculate distance using Haversine formula
    #                         distance_km_ndvi = haversine(lat_dust_source, lon_dust_source, lat_closest_ndvi,
    #                                                      lon_closest_ndvi)
    #                         values.append(distance_km_ndvi)
    #
    #                         # distance to Dense vegetation
    #                         # Find the indices of NDVI values within the range [0.5, 1]
    #                         valid_ndvi_indices_dense = np.where((ndvi_data > 0.5) & (ndvi_data <= 1))
    #
    #                         # Create a list of coordinates within the specified NDVI range
    #                         valid_ndvi_coords_dense = list(zip(valid_ndvi_indices_dense[0], valid_ndvi_indices_dense[1]))
    #
    #                         # Calculate distances between the current point and all valid NDVI coordinates
    #                         distances_ndvi_dense = cdist([point], valid_ndvi_coords_dense)
    #
    #                         # Find the index of the minimum distance
    #                         min_distance_index_ndvi_dense = np.argmin(distances_ndvi_dense)
    #
    #                         # Get the coordinates of the closest NDVI value in the specified range
    #                         closest_ndvi_coords_dense = valid_ndvi_coords_dense[min_distance_index_ndvi_dense]
    #
    #                         # Convert pixel coordinates to geographical coordinates
    #                         lon_closest_ndvi_dense, lat_closest_ndvi_dense = raster_transform * (
    #                             closest_ndvi_coords_dense[1], closest_ndvi_coords_dense[0])
    #
    #                         # Calculate distance using Haversine formula
    #                         distance_km_ndvi_dense = haversine(lat_dust_source, lon_dust_source, lat_closest_ndvi_dense,
    #                                                      lon_closest_ndvi_dense)
    #                         values.append(distance_km_ndvi_dense)
    #
    #
    #             values.append(row['ID'])
    #             values.append(year)
    #             values.append(point[0])
    #             values.append(point[1])
    #             temp_df = pd.DataFrame([values], columns=df.columns)
    #             dfs.append(temp_df)
    #
    #             # Check if the list of DataFrames is not empty before concatenating
    #     if dfs:
    #         df = pd.concat(dfs, ignore_index=True)
    #         # print(df.dtypes)


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

    # Identify categorical columns
    categorical_columns = ['landcover', 'soil_type']

    # Use OneHotEncoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    # Concatenate one-hot encoded columns with the original DataFrame
    df_encoded = pd.concat([df.drop(categorical_columns, axis=1),
                            pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))],
                           axis=1)

    # drop not important columns
    df_encoded = df_encoded.drop(
        # columns=['Water', 'Clay', 'Silt', 'Silt_Loam', 'Silt_Clay', 'Sand_Clay', 'Silt_Clay_Loam', 'Urban'])
        columns = ['landcover_Water', 'soil_type_Silt_Loam', 'landcover_Urban'])
    # 'soil_type_Silt_Clay_Loam'

    if (255 in df_encoded.values):
        # Drop rows containing -1 in any attribute
        df_encoded = df_encoded[df_encoded.apply(lambda row: 255 not in row.values, axis=1)]

    if (-1 in df_encoded.values):
        # Drop rows containing -1 in any attribute
        df_encoded = df_encoded[df_encoded.apply(lambda row: -1 not in row.values, axis=1)]

    if df_encoded.isna().any().any():
        # Drop rows containing NaN values in any attribute
        df_encoded = df_encoded.dropna()

    df_encoded.reset_index(drop=True, inplace=True)

    pk.dump(df_encoded, open(f'{dustsourcespickle}.pickle','wb'))


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
            if CalculateDistance:
                DistStr = 'Dist'
            else:
                DistStr = ''
            dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}_{DistStr}'

            createDatasetFunc(year_list_period, window_size,dustsourcespickle)
    else:
        print(f'CalculateSeasons is set to {CalculateSeasons}')
        PeriodName = len(year_list)
        print(f'Creating Data set for {year_list}')
        if CalculateDistance:
            DistStr = 'Dist'
        else:
            DistStr = ''
        dustsourcespickle = f'df_dustsources_WS{window_size}_X_{window_size}_PN{PeriodName}_SP_{statisticalParams}_{DistStr}'
        createDatasetFunc(year_list, window_size,dustsourcespickle)

###################################################################
# #### Load Data set ##############################################
###################################################################


def loadDatSet(df,window_size):
    print(f'windows size: {window_size}')

    # drop original categorical columns
    df = df.drop(columns=['X', 'Y','Year', 'landcover', 'soil_type'])

    df.to_csv(f'{dustsourcespickle}.csv', index=False)
