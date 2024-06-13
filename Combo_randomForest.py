import pickle as pk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.spatial.distance import pdist, squareform
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
import math
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import random
import seaborn as sns
from sklearn.preprocessing import binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold
import geopandas as gpd
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression

################### Configuration #####################
kernel = 'adaptive'  # adaptive or fixed
bw = 140  # Band Width for local models
mincatobs = 6  # minimum number of labels in each category for local models
LocalSoatialWeight = 'Bi-square'  # Bi-square or Gaussian
giveclassweight = True
givesampleweight = True
n_splits = 3  # cross validation folds
LocalCrossValidation = False  # Performs Cross validation on local models with n_splits
Case = 'Classification'  # Classification or Regression
importance_threshold = 1  # Removes less important features from Global Model
InBagSamples = 0.7  # from 0.5 to 0.9 for local models weighted bootstrapping
AnalyzeResiduales = True
################### Import the dataset #####################

local_param_grid = {
    'bw': list(range(135, 146)),
    'LocalSpatialWeight': ['Bi-square', 'Gaussian'],
    'n_splits': [3],
    'Case': ['Classification'],
    'importance_threshold': [0, 0.1, 0.2, 0.5, 1],
    'InBagSamples': [0.6, 0.65, 0.7],
    'givesampleweight': [True, False],
    'giveclassweight': [True, False]
}

from itertools import product

# Generate all combinations
combinations = list(product(*[v if isinstance(v, list) else [v] for v in local_param_grid.values()]))

os.chdir("/cephyr/users/alihaki/Alvis/Desktop/Duststorm/DustStormModeling/For training/")

dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP__'
# dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP__Dist_WS'


df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))

df = df.dropna()

df.reset_index(drop=True, inplace=True)

coords = df[['X', 'Y']]

######## No Dist
df = df.drop(columns=['Year', 'Profile_curvature', 'Plan_curvature', 'X', 'Y'])
Numerical_cols = ['Soil_evaporation', 'Precipitation', 'Soil_moisture', 'NDVI',
                  'Elevation', 'Aspect', 'Curvature', 'Distance_to_river', 'Slope', 'Wind_Speed']

Categorical_cols = ['Lakes', 'landcover_Cropland', 'landcover_Natural_vegetation',
                    'soil_type_Clay_Loam', 'soil_type_Loam', 'soil_type_Loam_Sand',
                    'soil_type_Sand', 'soil_type_Sand_Clay_Loam', 'soil_type_Sand_Loam',
                    'soil_type_Silt']

# Assuming 'dust_storm' is the column you want to exclude
columns_to_scale = [col for col in df.columns if col not in ['dust_storm', 'Lakes']]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the specified columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
dframe = df.copy()

X = df.drop(['dust_storm'], axis=1)
y = df['dust_storm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

best_score = 0
best_params = None
# Print all combinations
for combo in combinations:
    print('############### Next Grid search Iteration ############')
    bw, LocalSpatialWeight, n_splits, Case, importance_threshold, InBagSamples, givesampleweight, giveclassweight = combo  # Unpack the combination tuple

    # print(combo)

    if Case == 'Classification':
        print('############ Global model Metrics, binary classification #############')

        # if AnalyzeResiduales:
        #     # Initialize the logistic regression model
        #     LinearModel = LogisticRegression()
        #
        #     # Train the model on the training data
        #     LinearModel.fit(X_train, y_train)
        #
        #     # Make predictions on the testing data
        #     predictions = LinearModel.predict(X_test)
        #
        #     # Evaluate the model
        #     accuracy = accuracy_score(y_test, predictions)
        #     residual = y_test - predictions
        #
        #     # Find indices where residual is not equal to 0
        #     Nonlinearindices = np.where(residual != 0)[0]
        #     X_residuals = X_test.iloc[Nonlinearindices]
        #     y_residuals = y_test.iloc[Nonlinearindices]
        #     print('finish')

        # Additional stopping parameters
        params = {
            # Existing hyperparameters
            'n_estimators': 111,
            'max_depth': 12,
            'max_features': 12,
            'min_samples_split': 7,
            'min_samples_leaf': 3,
            'max_leaf_nodes': 99,
            'criterion': 'gini',
            'bootstrap': True,
            'min_impurity_decrease': 0.0013063066126301594,
            'ccp_alpha': 0.0030331248967434,
            'max_samples': 0.32352782230151134
        }

        Gl_Model = RandomForestClassifier(**params)
        Gl_Model.fit(X_train, y_train)

        y_pred = Gl_Model.predict(X_test)

        # Perform cross-validation
        cv_scores = cross_val_score(Gl_Model, X_train, y_train, cv=n_splits,
                                    scoring='accuracy')  # Change cv value as needed

        # Print the cross-validation scores
        print("Accuracy Cross-validation scores:", cv_scores)
        print("Mean CV Score fro Accuracy:", cv_scores.mean())

        # Calculate evaluation metrics
        conf_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        # Print evaluation metrics
        # Print the metrics
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Precision: {:.2f}%".format(precision * 100))
        print("Recall: {:.2f}%".format(recall * 100))
        print("F1-score: {:.2f}%".format(f1 * 100))
        print('Confusion matrix:\n True negative: %s \
                  \n False positive: %s \n False negative: %s \n True positive: %s'
              % (conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]))
        print('AUC: {:.2f}%'.format(auc * 100))

        feature_importances = Gl_Model.feature_importances_ * 100

        feature_names = X_train.columns

        # Sort features based on their importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_feature_importances = feature_importances[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # # Plotting
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(feature_importances)), sorted_feature_importances, align="center")
        # plt.xticks(range(len(feature_importances)), sorted_feature_names, rotation=45)
        # plt.xlabel("Feature")
        # plt.ylabel("Feature Importance (%)")
        # plt.title("Feature Importance Plot")
        # plt.tight_layout()
        # plt.show()

        # Identify columns with low importance
        low_importance_columns = X_train.columns[feature_importances < importance_threshold]

    elif Case == 'Regression':
        print('############ Global model Metrics, Regression #############')

        # if AnalyzeResiduales:
        #     # Initializing and fitting the linear regression model
        #     LinearModel = LinearRegression()
        #     LinearModel.fit(X, y)
        #
        #     # Making predictions on the testing set
        #     y_pred = LinearModel.predict(X)
        #
        #     # Calculating residuals
        #     residuals = y - y_pred
        #
        #     # Optionally, you can also calculate mean squared error (MSE) or other evaluation metrics
        #     mse = np.mean((y_pred - y) ** 2)
        #     print("Mean Squared Error:", mse)
        #
        #     plt.figure(figsize=(8, 6))
        #     plt.scatter(coords['X'], coords['Y'], c=residuals, cmap='coolwarm', alpha=0.7)
        #     plt.colorbar(label='Residuals')
        #     plt.xlabel('X')
        #     plt.ylabel('Y')
        #     plt.title('Residuals vs Coordinates')
        #     plt.grid(True)
        #     plt.show()

        # Additional stopping parameters for regression
        params = {
            # Existing hyperparameters
            'n_estimators': 75,
            'max_depth': 7,
            'max_features': 6,
            'max_leaf_nodes': 31,
            'min_samples_split': 8,
            'min_samples_leaf': 6,
            'bootstrap': True,
            'min_impurity_decrease': 0.0003043267155416363,
            'ccp_alpha': 0.0009282310679697315,
            'max_samples': 0.5,
            # Additional regression-specific parameters
            'criterion': 'squared_error'
        }

        Gl_Model = RandomForestRegressor(**params)
        Gl_Model.fit(X_train, y_train)

        y_pred = Gl_Model.predict(X_test)

        # Perform cross-validation
        cv_scores = cross_val_score(Gl_Model, X_train, y_train, cv=n_splits,
                                    scoring=make_scorer(mean_squared_error))  # Change cv value as needed

        # Print the cross-validation scores
        print("Mean Squared Error Cross-validation scores:", cv_scores)
        print("Mean CV Score for Mean Squared Error:", cv_scores.mean())

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Print evaluation metrics
        print("Mean Squared Error (MSE): {:.2f}".format(mse))
        print("Mean Absolute Error (MAE): {:.2f}".format(mae))
        print("R-squared (R2): {:.2f}".format(r2))

        threshold = 0.5
        y_pred_binary = (y_pred > threshold).astype(int)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        auc = roc_auc_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred_binary)

        # Print confusion matrix
        print('Confusion matrix:\n', conf_matrix)

        # Print evaluation metrics
        print("Accuracy: {:.2f}%".format(accuracy * 100))
        print("Precision: {:.2f}%".format(precision * 100))
        print("Recall: {:.2f}%".format(recall * 100))
        print("F1-score: {:.2f}%".format(f1 * 100))
        print('AUC: {:.2f}%'.format(auc * 100))

        feature_importances = Gl_Model.feature_importances_ * 100

        feature_names = X_train.columns

        # Sort features based on their importance
        sorted_indices = np.argsort(feature_importances)[::-1]
        sorted_feature_importances = feature_importances[sorted_indices]
        sorted_feature_names = [feature_names[i] for i in sorted_indices]

        # # Plotting
        # plt.figure(figsize=(10, 6))
        # plt.bar(range(len(feature_importances)), sorted_feature_importances, align="center")
        # plt.xticks(range(len(feature_importances)), sorted_feature_names, rotation=45)
        # plt.xlabel("Feature")
        # plt.ylabel("Feature Importance (%)")
        # plt.title("Feature Importance Plot")
        # plt.tight_layout()
        # plt.show()

        # Identify columns with low importance
        low_importance_columns = X_train.columns[feature_importances < importance_threshold]

    df_local = dframe.drop(low_importance_columns, axis=1)

    # Calculate pairwise distances
    distance_array = pdist(coords)

    # Convert the pairwise distances to a square matrix
    Dij = squareform(distance_array)

    if kernel == 'adaptive':
        Ne = bw
        print(f"Kernel: Adaptive\nNeighbours: {Ne}")
    elif kernel == 'fixed':
        print(f"Kernel: Fixed\nBandwidth: {bw}")


    def bootstrapWeighted(X_train, y_train, case_weights, mincatobs, InBagSamples, n_splits):
        # Calculate the number of samples to select
        num_samples = len(X_train)

        # Calculate the number of samples to be included in the bootstrap sample (70%)
        num_bootstrap_samples = int(InBagSamples * num_samples)

        # Select InBagSmples of indices with replacement based on case weights
        bootstrap_indices = np.random.choice(X_train.index, num_bootstrap_samples, replace=True,
                                             p=case_weights / np.sum(case_weights))

        oob_indices = np.setdiff1d(X_train.index, bootstrap_indices)

        # In bag X and y
        X_train_weighted = X_train.loc[bootstrap_indices]
        y_train_weighted = y_train.loc[bootstrap_indices]

        while 0 not in X_train_weighted['DNeighbour'].values:
            # Select the first InBagSamples of indices without replacement based on case weights
            bootstrap_indices = np.random.choice(X_train.index, num_bootstrap_samples, replace=True,
                                                 p=case_weights / np.sum(case_weights))
            # The remaining of indices are for OOB
            oob_indices = np.setdiff1d(X_train.index, bootstrap_indices)

            # In bag X and y
            X_train_weighted = X_train.loc[bootstrap_indices]
            y_train_weighted = y_train.loc[bootstrap_indices]

        value_counts = y_train_weighted.value_counts()
        min_category = y_train.value_counts().idxmin()
        if len(value_counts) < 2 or y_train_weighted.value_counts().get(min_category, 0) < n_splits + 1:
            # Ensure that the indices to remove contain rows with max_category
            min_category_indices = np.random.choice(y_train[y_train == min_category].index, n_splits + 1, replace=False)

            # Append min_category_indices to X_train_weighted and y_train_weighted
            X_train_weighted = pd.concat([X_train_weighted, X_train.loc[min_category_indices]])
            y_train_weighted = pd.concat([y_train_weighted, y_train.loc[min_category_indices]])

            # Remove min_category_indices from oob_indices
            oob_indices = np.setdiff1d(oob_indices, min_category_indices)

            # Add min_category_indices to bootstrap_indices
            bootstrap_indices = np.concatenate((bootstrap_indices, min_category_indices))

        # Save out-of-bag samples
        X_OOB = X_train.loc[oob_indices]
        y_OOB = y_train.loc[oob_indices]

        # Onlu in Bag Spatial Weighted
        spatial_weights = case_weights.loc[bootstrap_indices]

        # Weighted Bootstrapping outputs
        return X_train_weighted, y_train_weighted, X_OOB, y_OOB, spatial_weights


    obs = len(df_local)

    # Add 'pointID' column to 'dframe'
    df_local['pointID'] = range(0, len(df_local))
    coords['pointID'] = range(0, len(df_local))

    prediction_df = pd.DataFrame(columns=['PointID', 'y_test_l', 'y_pred_l', 'TruePositive',
                                          'TrueNegative', 'FalsePositive', 'FalseNegative'])

    Validation_OOB_df = pd.DataFrame(columns=['PointID', 'y_OOB', 'y_validation_OOB', 'TruePositive',
                                              'TrueNegative', 'FalsePositive', 'FalseNegative'])

    layered_df_dict = {}
    local_models = {}
    Feature_Importance_l = {}
    local_model_accuracy_l = {}
    Local_models_cv = []

    for m in range(0, obs):
        # Get the data
        DNeighbour = Dij[:, m]

        #### Create a new DataFrame 'DataSet' with 'df_local' and 'DNeighbour'
        DataSet = df_local.copy()
        DataSet['DNeighbour'] = DNeighbour
        # Sort by distance
        DataSetSorted = DataSet.sort_values(by='DNeighbour')
        if kernel == 'adaptive':
            cc = 1
            # Keep Nearest Neighbours
            SubSet = DataSetSorted.iloc[:Ne, :]
        elif kernel == 'fixed':
            SubSet = DataSetSorted[DataSetSorted['DNeighbour'] <= bw]
            Kernel_H = bw

        # Make sure there is at least one type of both labels in the subset
        while len(SubSet['dust_storm'].unique()) < 2 or SubSet['dust_storm'].value_counts()[0] < mincatobs or \
                SubSet['dust_storm'].value_counts()[1] < mincatobs:
            SubSet = DataSetSorted.iloc[:Ne + cc, :]
            cc += 1
        if cc > 1:
            # before_removal_coords = pd.merge(SubSet, coords, on='pointID', how='left')

            # Plot KDE plot
            # plt.figure(figsize=(10, 6))
            # sns.kdeplot(data=SubSet, x='DNeighbour', fill=True)
            # plt.title('Probability Distribution of DNeighbour')
            # plt.xlabel('DNeighbour')
            # plt.ylabel('Probability Density')
            # plt.show()

            value_counts = SubSet['dust_storm'].value_counts()
            max_category = value_counts.idxmax()

            # Calculate the probabilities based on 'DNeighbour'
            probabilities = SubSet['DNeighbour'] / SubSet['DNeighbour'].sum()

            # Ensure that the indices to remove contain rows with max_category
            max_category_indices = SubSet[SubSet['dust_storm'] == max_category].index
            # Generate random indices with higher probability for higher 'DNeighbour' values
            remove_indices = np.random.choice(max_category_indices,
                                              size=len(max_category_indices) - (bw - mincatobs),
                                              replace=False,
                                              p=probabilities.loc[max_category_indices] /
                                                probabilities.loc[max_category_indices].sum())

            # Remove the selected rows
            SubSet = SubSet.drop(remove_indices)

            # Plot KDE plot
            # plt.figure(figsize=(10, 6))
            # sns.kdeplot(data=SubSet, x='DNeighbour', fill=True)
            # plt.title('Probability Distribution of DNeighbour')
            # plt.xlabel('DNeighbour')
            # plt.ylabel('Probability Density')
            # plt.show()

        if kernel == 'adaptive':
            Kernel_H = SubSet['DNeighbour'].max()
        elif kernel == 'fixed':
            Kernel_H = bw

        after_removal_coords = pd.merge(SubSet, coords, on='pointID', how='left')
        layered_df_dict[m] = after_removal_coords

        # # Plotting latitude and longitude coordinates
        # plt.figure(figsize=(10, 8))
        # plt.scatter(after_removal_coords['X'], after_removal_coords['Y'], c=after_removal_coords['dust_storm'],
        #             cmap='coolwarm', s=50, alpha=0.7, label='dust_storm')
        # plt.scatter(after_removal_coords['X'].iloc[0], after_removal_coords['Y'].iloc[0], color='red', marker='*', s=200,
        #             label='First Point')
        # plt.title('Scatter Plot of Latitude and Longitude Coordinates')
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # plt.grid(True)
        # plt.legend()
        # plt.show()

        #### Split training and test data for the local models
        X_l = SubSet.drop(['dust_storm'], axis=1)
        y_l = SubSet['dust_storm']
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.2, random_state=0,
                                                                    stratify=y_l)

        # Make sure there is at least one type of both labels in the Training data
        # value_counts = y_train_l.value_counts()
        while len(y_train_l.unique()) < 2 or 0 not in X_train_l['DNeighbour'].values or y_train_l.value_counts().values[
            y_train_l.value_counts().idxmin()] < math.floor(0.8 * mincatobs):
            random_integer = random.randint(0, 123)
            X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_l, y_l, test_size=0.2,
                                                                        random_state=random_integer, stratify=y_l)

        # Bi-square weights
        if LocalSoatialWeight == 'Bi-square':
            Wts_train = (1 - (X_train_l['DNeighbour'] / Kernel_H) ** 2) ** 2
        elif LocalSoatialWeight == 'Guassian':
            Wts_train = np.exp(-(X_train_l['DNeighbour'] ** 2) / (2 * Kernel_H ** 2))

        #### Use bootstrapWeighted to get weighted samples
        (X_train_l_weighted, y_train_l_weighted,
         X_OOB, y_OOB, case_weights) = bootstrapWeighted(X_train_l,
                                                         y_train_l,
                                                         Wts_train,
                                                         mincatobs,
                                                         InBagSamples,
                                                         n_splits)

        # Drop pointID
        # X_train_l_main_noPID = X_train_l.drop(['pointID'], axis=1)
        X_train_l_noPID = X_train_l_weighted.drop(['pointID', 'DNeighbour'], axis=1)
        X_test_l_noPID = X_test_l.drop(['pointID', 'DNeighbour'], axis=1)
        X_OOB_noPID = X_OOB.drop(['pointID', 'DNeighbour'], axis=1)
        X_train_l_main_noPID = X_train_l.drop(['pointID', 'DNeighbour'], axis=1)

        if Case == 'Classification':
            #### Class Weights
            # Calculate class weights
            num_class_0_train = sum(y_train_l_weighted == 0)
            num_class_1_train = sum(y_train_l_weighted == 1)
            num_class_total = num_class_0_train + num_class_1_train

            weight_class_0_train = (num_class_total / (2 * num_class_0_train)) * 100
            weight_class_1_train = (num_class_total / (2 * num_class_1_train)) * 100

            weight_class_0_normalized = weight_class_0_train / (weight_class_0_train + weight_class_1_train)
            weight_class_1_normalized = weight_class_1_train / (weight_class_0_train + weight_class_1_train)

            # Create a dictionary of class weights
            # class_weights = {0: weight_class_0_normalized, 1: weight_class_1_normalized}
            class_weights = {0: weight_class_0_train, 1: weight_class_1_train}
            if giveclassweight:
                params['class_weight'] = class_weights
            else:
                params['class_weight'] = None
            params['bootstrap'] = False
            params['max_samples'] = None

            case_weights_float = case_weights.astype(float)
            #### Randomforest classifier
            LO_Model = RandomForestClassifier(**params)
            # FIT THE MODEL TO THE TRAINING DATA
            # , sample_weight = case_weights_float
            if givesampleweight:
                LO_Model.fit(X_train_l_noPID, y_train_l_weighted, sample_weight=case_weights_float)
            else:
                LO_Model.fit(X_train_l_noPID, y_train_l_weighted)
            # LO_Model.fit(X_train_l_main_noPID, y_train_l,sample_weight = Wts_train)

            #### TEST PREDICTION
            y_pred_l = LO_Model.predict(X_test_l_noPID)
            local_model_accuracy = accuracy_score(y_test_l, y_pred_l)
            local_models[m] = LO_Model
            if LocalCrossValidation:
                # Perform cross-validation
                cv_scores = cross_val_score(LO_Model, X_train_l_noPID, y_train_l_weighted, cv=n_splits,
                                            scoring='accuracy')
                prediction_cv = cv_scores.mean()
                Local_models_cv.append(prediction_cv)
            else:
                prediction_cv = 0
                Local_models_cv.append(prediction_cv)
        elif Case == 'Regression':
            params['bootstrap'] = False
            params['max_samples'] = None
            LO_Model = RandomForestRegressor(**params)
            # FIT THE MODEL TO THE TRAINING DATA
            # , sample_weight = case_weights_float
            LO_Model.fit(X_train_l_noPID, y_train_l_weighted)
            # LO_Model.fit(X_train_l_main_noPID, y_train_l,sample_weight = Wts_train)

            #### TEST PREDICTION
            y_pred_regression = LO_Model.predict(X_test_l_noPID)
            y_pred_l = (y_pred_regression > threshold).astype(int)
            local_model_accuracy = accuracy_score(y_test_l, y_pred_l)

            local_models[m] = LO_Model

            if LocalCrossValidation:
                # Perform cross-validation
                cv_scores = cross_val_score(LO_Model, X_train_l_noPID, y_train_l_weighted, cv=n_splits,
                                            scoring=make_scorer(mean_squared_error))
                prediction_cv = cv_scores.mean()
                Local_models_cv.append(prediction_cv)
            else:
                prediction_cv = 0
                Local_models_cv.append(prediction_cv)

        feature_importances_local = LO_Model.feature_importances_ * 100
        Feature_Importance_l[m] = feature_importances_local
        local_model_accuracy_l[m] = local_model_accuracy

        prediction_row = pd.DataFrame({'PointID': X_test_l['pointID'],
                                       'y_test_l': y_test_l,
                                       'y_pred_l': y_pred_l,
                                       'TruePositive': 0,
                                       'TrueNegative': 0,
                                       'FalsePositive': 0,
                                       'FalseNegative': 0}, )

        prediction_df = pd.concat([prediction_df, prediction_row], ignore_index=True)

        #### OUT OF BAG VALDIATION
        if Case == 'Classification':
            y_validation_OOB = LO_Model.predict(X_OOB_noPID)
        elif Case == 'Regression':
            y_validation_OOB_regression = LO_Model.predict(X_OOB_noPID)
            y_validation_OOB = (y_validation_OOB_regression > threshold).astype(int)

        validation_OOB_row = pd.DataFrame({'PointID': X_OOB['pointID'],
                                           'y_OOB': y_OOB,
                                           'y_validation_OOB': y_validation_OOB,
                                           'TruePositive': 0,
                                           'TrueNegative': 0,
                                           'FalsePositive': 0,
                                           'FalseNegative': 0}, )
        Validation_OOB_df = pd.concat([Validation_OOB_df, validation_OOB_row], ignore_index=True)


    print('############ Local model Metrics PREDICTION #############')

    # Calculate True Positive, True Negative, False Positive, False Negative for each row
    for index, row in prediction_df.iterrows():
        if row['y_test_l'] == 1 and row['y_pred_l'] == 1:
            prediction_df.at[index, 'TruePositive'] = 1
        elif row['y_test_l'] == 0 and row['y_pred_l'] == 0:
            prediction_df.at[index, 'TrueNegative'] = 1
        elif row['y_test_l'] == 0 and row['y_pred_l'] == 1:
            prediction_df.at[index, 'FalsePositive'] = 1
        elif row['y_test_l'] == 1 and row['y_pred_l'] == 0:
            prediction_df.at[index, 'FalseNegative'] = 1

    # Group by PointID and sum the values for each group
    aggregated_df = prediction_df.groupby('PointID').agg({
        'TruePositive': 'sum',
        'TrueNegative': 'sum',
        'FalsePositive': 'sum',
        'FalseNegative': 'sum'
    }).reset_index()

    aggregated_df['MajorityClass'] = aggregated_df[
        ['TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative']].idxmax(axis=1)

    aggregated_df['MajorityClass'].value_counts().get('FalseNegative', 0)
    # Calculate counts for TruePositive, TrueNegative, FalsePositive, and FalseNegative
    tp_count = aggregated_df['MajorityClass'].value_counts().get('TruePositive', 0)
    tn_count = aggregated_df['MajorityClass'].value_counts().get('TrueNegative', 0)
    fp_count = aggregated_df['MajorityClass'].value_counts().get('FalsePositive', 0)
    fn_count = aggregated_df['MajorityClass'].value_counts().get('FalseNegative', 0)

    # Create a confusion matrix DataFrame
    confusion_matrix_local = pd.DataFrame({
        'Predicted_Positive': [tp_count, fp_count],
        'Predicted_Negative': [fn_count, tn_count]
    }, index=['Actual_Positive', 'Actual_Negative'])

    # Print or display the confusion matrix
    print('Confusion Matrix for Prediction')
    print(confusion_matrix_local)

    TP_L = tp_count
    TN_L = tn_count
    FP_L = fp_count
    FN_L = fn_count

    # Calculate performance metrics
    accuracy_local = (TP_L + TN_L) / (TP_L + TN_L + FP_L + FN_L)
    precision_local = TP_L / (TP_L + FP_L)
    recall_local = TP_L / (TP_L + FN_L)
    f1_local = 2 * precision_local * recall_local / (precision_local + recall_local)

    local_CV_score = np.mean(Local_models_cv)

    # Print or display the calculated metrics
    print("Accuracy: {:.2f}".format(accuracy_local * 100))
    print("Precision: {:.2f}".format(precision_local * 100))
    print("Recall: {:.2f}".format(recall_local * 100))
    print("F1 Score: {:.2f}".format(f1_local * 100))
    print(f"{n_splits} fold cross validation score = {local_CV_score}")

    print('################ finish ################')
    # Track the best performing combination
    if accuracy_local > best_score:
        best_score = accuracy_local
        best_params = combo

print(best_score)
print(best_params)
