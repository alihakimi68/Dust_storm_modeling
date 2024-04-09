import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import pdist, squareform
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
import os

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import randint
import pickle as pk
import random

Model = 'XGB'
HyperTune = False

os.chdir("/For training/")

# For df_dustsources_WS0_X_0_PN20_SP_
# dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP_'
# FeatureCount = 20
# Estimator = 400

# For df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod
dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod'
# FeatureCount = 25
# Estimator = 400
# For df_dustsources_WS7_X_7_PN20_SP_WMe_
# dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_WMe_'
# FeatureCount = 15
df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))

# drop original categorical columns
# df = df.drop(columns=['X', 'Y','Year', 'landcover', 'soil_type'])

# Check and remove NA values
df = df.dropna()

columns_to_keep = ['Soil_evaporation variance', 'Soil_evaporation median', 'Lakes entropy',
                   'Lakes mode', 'Precipitation variance', 'Precipitation median',
                   'Soil_moisture variance', 'Soil_moisture median', 'NDVI variance','NDVI median',
                   'Elevation variance', 'Elevation median','Aspect variance', 'Aspect median',
                   'Curvature variance','Curvature median', 'Plan_curvature variance',
                   'Plan_curvature median','Profile_curvature variance', 'Profile_curvature median',
                   'Distance_to_river variance', 'Distance_to_river median','Slope variance',
                   'Slope median', 'Bare_Soil', 'Cropland','Natural_vegetation', 'Clay_Loam',
                   'Loam', 'Loam_Sand', 'Sand','Sand_Clay_Loam', 'Sand_Loam','dust_storm','X', 'Y']

df = df.loc[:, columns_to_keep]

# Assuming 'dust_storm' is the column you want to exclude
columns_to_scale = [col for col in df.columns if col not in ['dust_storm', 'X', 'Y']]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the specified columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# Drop 'Year' column
#df = df.drop(columns=['Year'])

train_valid_df = df.copy()
coords = train_valid_df[['X', 'Y']]

# Drop the specified coordinates from the train data frame
train_valid_df = train_valid_df.drop(columns=['X', 'Y'])

# Use train_test_split to create indices for random sampling
X = train_valid_df.drop(['dust_storm'], axis=1)
y = train_valid_df['dust_storm']
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2,
                                                                  random_state=42, stratify=y)

if HyperTune:
    param_dist = {
        'n_estimators': randint(600, 1500),
        'max_features': ['sqrt', 'log2'] + list(range(6, 13)),
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6],
        'criterion': ['entropy', 'gini'],
        'bootstrap': [True, False]
    }
    # Create a random forest classifier
    rf_classifier = RandomForestClassifier()

    # Use RandomizedSearchCV for hyperparameter tuning
    random_search = RandomizedSearchCV(
        rf_classifier,
        param_distributions=param_dist,
        n_iter=10,  # Adjust the number of iterations as needed
        cv=5,  # Adjust the number of cross-validation folds as needed
        scoring='accuracy',  # Use an appropriate scoring metric
        random_state=42,
        n_jobs=-1  # Use all available CPU cores
    )

    # Fit the random search to the data
    random_search.fit(X_train, y_train)

    # Print the best hyperparameters
    print("Best Hyperparameters:", random_search.best_params_)

    # Get the best model from the random search
    best_rf_model = random_search.best_estimator_

    # Evaluate the best model on the test set
    accuracy = best_rf_model.score(X_test, y_test)
    print("Test Accuracy:", accuracy)
# Make a copy of the dataset for the global model
dframe_full = train_valid_df.copy()

# Count the number of observations in the data
obs = len(dframe_full)
params = {}
if Model == 'RF':
    params['n_estimators'] = 200
    params['max_depth'] = 8
    params['max_features'] = 8
    params['random_state'] = 42
    params['criterion'] = 'gini'
    params['ccp_alpha'] = 0.012034044024305203
    # params['max_leaf_nodes'] = 20
    params['max_samples'] = 0.32352782230151134
    params['min_impurity_decrease'] = 0.011134562990795117
    params['min_samples_split'] = 8
    params['min_samples_leaf'] = 2
    # params['min_weight_fraction_leaf'] = 0.00944043928811733
    params['n_jobs'] = -1

    # params['bootstrap'] = True


    Gl_Model = RandomForestClassifier(**params)
else:
    params['objective'] = 'binary:logistic'
    params['num_class'] = 1
    params['eval_metric'] = 'auc'
    params['learning_rate'] = 0.0980587910917366
    params['max_depth'] = 7
    params['min_child_weight'] = 1
    params['n_estimators'] = 695
    params['colsample_bytree'] = 0.8523149652539629
    params['subsample'] = 0.8976451164881573
    # params['gamma'] = 0.1
    # params['num_parallel_tree'] = 2

    Gl_Model = xgb.XGBClassifier(**params)

Gl_Model.fit(X_train, y_train)
y_pred = Gl_Model.predict(X_test)

print('############ Global model Metrics #############')

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

y_all_predict = Gl_Model.predict(X)

print('############ Global model Metrics for all Observations #############')

accuracy_all = accuracy_score(y, y_all_predict)
precision_all = precision_score(y, y_all_predict)
recall_all = recall_score(y, y_all_predict)
f1_all = f1_score(y, y_all_predict)
conf_matrix_all = confusion_matrix(y, y_all_predict)
auc_all = roc_auc_score(y, y_all_predict)

# Print the metrics
print("Accuracy: {:.2f}%".format(accuracy_all * 100))
print("Precision: {:.2f}%".format(precision_all * 100))
print("Recall: {:.2f}%".format(recall_all * 100))
print("F1-score: {:.2f}%".format(f1_all * 100))
print('Confusion matrix:\n True negative: %s \
          \n False positive: %s \n False negative: %s \n True positive: %s'
      % (conf_matrix_all[0, 0], conf_matrix_all[0, 1], conf_matrix_all[1, 0], conf_matrix_all[1, 1]))
print('AUC: {:.2f}%'.format(auc_all * 100))


# Global Model Feature Importance
feature_importances = Gl_Model.feature_importances_ * 100

feature_names = X_train.columns

# Sort features based on their importance
sorted_indices = np.argsort(feature_importances)[::-1]
sorted_feature_importances = feature_importances[sorted_indices]
sorted_feature_names = [feature_names[i] for i in sorted_indices]

# Plotting
# plt.figure(figsize=(10, 6))
# plt.bar(range(len(feature_importances)), sorted_feature_importances, align="center")
# plt.xticks(range(len(feature_importances)), sorted_feature_names, rotation=45)
# plt.xlabel("Feature")
# plt.ylabel("Feature Importance (%)")
# plt.title("Feature Importance Plot")
# plt.tight_layout()
# plt.show()

# Set the importance threshold
importance_threshold = 0

# Identify columns with low importance
low_importance_columns = X_train.columns[feature_importances < importance_threshold]
# dframe = dframe_full.drop(low_importance_columns, axis=1)
# dframe = dframe_full.drop(['Lakes', 'Cropland', 'Natural_vegetation', 'Clay_Loam','Sand_Clay_Loam'], axis=1)
dframe = dframe_full
# Calculate pairwise distances
distance_array = pdist(coords)

# Convert the pairwise distances to a square matrix
Dij = squareform(distance_array)

kernel = 'adaptive'
bw = 140
mincatobs = 6
if kernel == 'adaptive':
    Ne = bw
    print(f"Kernel: Adaptive\nNeighbours: {Ne}")
elif kernel == 'fixed':
    print(f"Kernel: Fixed\nBandwidth: {bw}")

prediction_df = pd.DataFrame(columns=['PointID', 'y_test_l', 'y_pred_l', 'TruePositive',
                                      'TrueNegative', 'FalsePositive', 'FalseNegative'])

Validation_OOB_df = pd.DataFrame(columns=['PointID', 'y_OOB', 'y_validation_OOB', 'TruePositive',
                                      'TrueNegative', 'FalsePositive', 'FalseNegative'])

prediction_row_2nd_df = pd.DataFrame(columns=['PointID', 'y_test_l', 'y_pred_l'])

results_data = []

def bootstrapWeighted(X_train, y_train, case_weights):
    # Calculate the number of samples to select
    num_samples = len(X_train)

    # Calculate the number of samples to be included in the bootstrap sample (70%)
    num_bootstrap_samples = int(0.7 * num_samples)

    # Select the first 80% of indices without replacement based on case weights
    bootstrap_indices = np.random.choice(num_samples, num_bootstrap_samples, replace=True,
                                         p=case_weights / np.sum(case_weights))

    # The remaining 20% of indices are for OOB
    oob_indices = np.setdiff1d(np.arange(num_samples), bootstrap_indices)

    # Apply weights to training samples
    X_train_weighted = X_train.iloc[bootstrap_indices]
    y_train_weighted = y_train.iloc[bootstrap_indices]

    while len(y_train_weighted.unique()) < 2:
        # Select the first 80% of indices without replacement based on case weights
        bootstrap_indices = np.random.choice(num_samples, num_bootstrap_samples, replace=True,
                                             p=case_weights / np.sum(case_weights))
        # The remaining 20% of indices are for OOB
        oob_indices = np.setdiff1d(np.arange(num_samples), bootstrap_indices)

        # Apply weights to training samples
        X_train_weighted = X_train.iloc[bootstrap_indices]
        y_train_weighted = y_train.iloc[bootstrap_indices]

    # case_weights = case_weights.iloc[bootstrap_indices]

    # Save out-of-bag samples
    X_OOB = X_train.iloc[oob_indices]
    y_OOB = y_train.iloc[oob_indices]

    # Identify and count duplicates based on 'Point_ID'
    duplicate_counts = X_train_weighted['pointID'].value_counts()

    # Remove duplicates based on 'Point_ID'
    X_train_unique = X_train_weighted.drop_duplicates('pointID')
    X_train_indices = X_train_unique.index

    # y_train_weighted = y_train_weighted[~y_train_weighted.index.duplicated(keep='first')]

    # Reindex y_train_weighted
    # y_train_unique = y_train_weighted.reindex(X_train_indices)

    # case_weights = case_weights.loc[X_train_indices]

    # Create a weight list based on the number of duplicates
    duplicate_weights = X_train_unique['pointID'].map(duplicate_counts)
    duplicate_weights = X_train_weighted['pointID'].map(duplicate_counts)

    # Sample_eights = case_weights * duplicate_weights

    # return X_train_unique, y_train_unique, X_OOB, y_OOB, duplicate_weights
    return X_train_weighted, y_train_weighted, X_OOB, y_OOB, duplicate_weights


for m in range(0,obs):
    # Get the data
    DNeighbour = Dij[:, m]

    # Add 'pointID' column to 'dframe'
    dframe['pointID'] = range(0, len(dframe))
    coords['pointID'] = range(0, len(dframe))

    # Create a new DataFrame 'DataSet' with 'dframe' and 'DNeighbour'
    DataSet = dframe.copy()
    DataSet['DNeighbour'] = DNeighbour

    # Sort by distance
    DataSetSorted = DataSet.sort_values(by='DNeighbour')
    if kernel == 'adaptive':

        cc = 1
        # Keep Nearest Neighbours
        SubSet = DataSetSorted.iloc[:Ne, :]

        # Make sure there is at least one type of both labels in the subset
        while len(SubSet['dust_storm'].unique()) < 2 or any(SubSet['dust_storm'].value_counts() < mincatobs):
            SubSet = DataSetSorted.iloc[:Ne + cc, :]
            cc += 1
        if cc > 1 :

            # before_removal_coords = pd.merge(SubSet, coords, on='pointID', how='left')

            value_counts = SubSet['dust_storm'].value_counts()
            max_category = value_counts.idxmax()

            # Calculate the probabilities based on 'DNeighbour'
            probabilities = SubSet['DNeighbour'] / SubSet['DNeighbour'].sum()

            # Ensure that the indices to remove contain rows with max_category
            max_category_indices = SubSet[SubSet['dust_storm'] == max_category].index
            # Generate random indices with higher probability for higher 'DNeighbour' values
            remove_indices = np.random.choice(max_category_indices, size=len(max_category_indices) - (bw -mincatobs), replace=False,
                                              p=probabilities.loc[max_category_indices] / probabilities.loc[
                                                  max_category_indices].sum())

            # Remove the selected rows
            SubSet = SubSet.drop(remove_indices)

        Kernel_H = SubSet['DNeighbour'].max()
    elif kernel == 'fixed':
        SubSet = DataSetSorted[DataSetSorted['DNeighbour'] <= bw]
        Kernel_H = bw

    # after_removal_coords = pd.merge(SubSet, coords, on='pointID', how='left')
    # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.scatter(before_removal_coords['X'], before_removal_coords['Y'], color='blue',
    #             label='Before Removal')
    # plt.scatter(after_removal_coords['X'], after_removal_coords['Y'], color='red', label='After Removal')
    # plt.title('Coords pointID Before and After Removal')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.legend()
    # plt.show()
    ######################################

    # Split training and test data
    train_indices = np.random.choice(len(SubSet['dust_storm']), size=int(0.8 * len(SubSet['dust_storm'])), replace=False)

    SubSet_train = SubSet.iloc[train_indices]
    train_indices_np = np.array(train_indices)
    valid_indices = np.setdiff1d(np.arange(len(SubSet)), train_indices_np)

    SubSet_valid = SubSet.iloc[valid_indices]

    X_train_l = SubSet_train.drop(['dust_storm'], axis=1)
    X_test_l = SubSet_valid.drop(['dust_storm'], axis=1)
    y_train_l = SubSet_train['dust_storm']
    y_test_l = SubSet_valid['dust_storm']

    # Make sure there is at least one type of both labels in the Training data
    while len(y_train_l.unique()) < 2:
        random_integer = random.randint(15, 45)
        train_indices = np.random.choice(len(SubSet['dust_storm']), size=int(0.8 * len(SubSet['dust_storm'])),
                                         replace=False)

        SubSet_train = SubSet.iloc[train_indices]
        train_indices_np = np.array(train_indices)
        valid_indices = np.setdiff1d(np.arange(len(SubSet)), train_indices_np)

        SubSet_valid = SubSet.iloc[valid_indices]

        X_train_l = SubSet_train.drop(['dust_storm'], axis=1)
        X_test_l = SubSet_valid.drop(['dust_storm'], axis=1)
        y_train_l = SubSet_train['dust_storm']
        y_test_l = SubSet_valid['dust_storm']

    #####################################

    # # Use train_test_split to create indices for random sampling
    # Xl = SubSet.drop(['dust_storm'], axis=1)
    # # Xl = SubSet.drop(['Lakes', 'Cropland', 'Natural_vegetation', 'Clay_Loam'], axis=1)
    # yl = SubSet['dust_storm']
    # X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(Xl, yl, test_size=0.2,
    #                                                     random_state=42)
    #
    # while len(y_train_l.unique()) < 2 or len(y_test_l.unique()) < 2 or any(y_train_l.value_counts() < 1) or any(
    #         y_test_l.value_counts() < 1):
    #     random_integer = random.randint(15, 45)
    #     X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(Xl, yl, test_size=0.2,
    #                                                                 random_state=random_integer)

    # Bi-square weights
    Wts_train = (1 - (X_train_l['DNeighbour'] / Kernel_H) ** 2) ** 2
    # Gaussian weights
    # Wts_train = np.exp(-(X_train_l['DNeighbour']**2) / (2 * Kernel_H**2))

    # Use bootstrapWeighted to get weighted samples
    X_train_l_weighted, y_train_l_weighted, X_OOB, y_OOB , case_weights = bootstrapWeighted(X_train_l,
                                                                             y_train_l, Wts_train)

    # Drop pointID
    X_train_l_noPID = X_train_l_weighted.drop(['pointID'], axis=1)
    X_test_l_noPID = X_test_l.drop(['pointID'], axis=1)
    X_OOB_noPID = X_OOB.drop(['pointID'], axis=1)

    # # Sort X_train_l_noPID and y_train_l_weighted based on 'DNeighbour'
    # train_sort_index = X_train_l_noPID['DNeighbour'].sort_values().index
    # X_train_l_noPID = X_train_l_noPID.loc[train_sort_index]
    # y_train_l_weighted = y_train_l_weighted.loc[train_sort_index]
    # case_weights = case_weights.loc[train_sort_index]
    #
    # # Sort X_test_l_noPID and y_test_l based on 'DNeighbour'
    # test_sort_index = X_test_l_noPID['DNeighbour'].sort_values().index
    # X_test_l_noPID = X_test_l_noPID.loc[test_sort_index]
    # y_test_l = y_test_l.loc[test_sort_index]
    #
    # # Sort X_OOB_noPID and y_test_l based on 'DNeighbour'
    # test_sort_index = X_OOB_noPID['DNeighbour'].sort_values().index
    # X_OOB_noPID = X_OOB_noPID.loc[test_sort_index]
    # y_OOB = y_OOB.loc[test_sort_index]

    # Drop DNeighbour
    # X_train_l_noPID = X_train_l_noPID.drop(['DNeighbour'], axis=1)
    # X_test_l_noPID = X_test_l_noPID.drop(['DNeighbour'], axis=1)
    # X_OOB_noPID = X_OOB_noPID.drop(['DNeighbour'], axis=1)

    if Model == 'RF':

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

        params['class_weight'] = class_weights
        # params['bootstrap'] = True
        LO_Model = RandomForestClassifier(**params)

    else:
        # params['class_weight'] = 'balanced'
        params['n_jobs'] = -1
        LO_Model = xgb.XGBClassifier(**params)

    # FIT THE MODEL TO THE TRAINING DATA
    case_weights_float = case_weights.astype(float)

    LO_Model.fit(X_train_l_noPID, y_train_l_weighted, sample_weight = case_weights_float)

    ##### TEST PREDICTION ####
    y_pred_l = LO_Model.predict(X_test_l_noPID)


    prediction_row = pd.DataFrame({'PointID': X_test_l['pointID'],
                                   'y_test_l': y_test_l,
                                   'y_pred_l': y_pred_l,
                                   'TruePositive': 0,
                                   'TrueNegative': 0,
                                   'FalsePositive': 0,
                                   'FalseNegative': 0}, )

    prediction_df = pd.concat([prediction_df, prediction_row], ignore_index=True)

    ##### OUT OF BAG VALDIATION ####
    y_validation_OOB = LO_Model.predict(X_OOB_noPID)


    validation_OOB_row = pd.DataFrame({'PointID': X_OOB['pointID'],
                                   'y_OOB': y_OOB,
                                   'y_validation_OOB': y_validation_OOB,
                                   'TruePositive': 0,
                                   'TrueNegative': 0,
                                   'FalsePositive': 0,
                                   'FalseNegative': 0}, )
    Validation_OOB_df = pd.concat([Validation_OOB_df, validation_OOB_row], ignore_index=True)

    prediction_row_2nd = pd.DataFrame({'PointID': [X_test_l['pointID'].iloc[0]],
                                       'y_test_l': [y_test_l.iloc[0]],
                                       'y_pred_l': [y_pred_l[0]]})
    prediction_row_2nd_df = pd.concat([prediction_row_2nd_df, prediction_row_2nd], ignore_index=True)

    print(m)

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

aggregated_df['MajorityClass'] = aggregated_df[['TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative']].idxmax(axis=1)


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

# Print or display the calculated metrics
print("Accuracy: {:.2f}".format(accuracy_local*100))
print("Precision: {:.2f}".format(precision_local*100))
print("Recall: {:.2f}".format(recall_local*100))
print("F1 Score: {:.2f}".format(f1_local*100))


print('############ Local model Metrics OOB VLIDATION #############')

# Calculate True Positive, True Negative, False Positive, False Negative for each row
for index, row in Validation_OOB_df.iterrows():
    if row['y_OOB'] == 1 and row['y_validation_OOB'] == 1:
        Validation_OOB_df.at[index, 'TruePositive'] = 1
    elif row['y_OOB'] == 0 and row['y_validation_OOB'] == 0:
        Validation_OOB_df.at[index, 'TrueNegative'] = 1
    elif row['y_OOB'] == 0 and row['y_validation_OOB'] == 1:
        Validation_OOB_df.at[index, 'FalsePositive'] = 1
    elif row['y_OOB'] == 1 and row['y_validation_OOB'] == 0:
        Validation_OOB_df.at[index, 'FalseNegative'] = 1

# Group by PointID and sum the values for each group
aggregated_oob_df = Validation_OOB_df.groupby('PointID').agg({
    'TruePositive': 'sum',
    'TrueNegative': 'sum',
    'FalsePositive': 'sum',
    'FalseNegative': 'sum'
}).reset_index()

aggregated_oob_df['MajorityClass'] = aggregated_oob_df[['TruePositive', 'TrueNegative', 'FalsePositive', 'FalseNegative']].idxmax(axis=1)

# Calculate counts for TruePositive, TrueNegative, FalsePositive, and FalseNegative
tp_count_OOB = aggregated_oob_df['MajorityClass'].value_counts().get('TruePositive', 0)
tn_count_OOB = aggregated_oob_df['MajorityClass'].value_counts().get('TrueNegative', 0)
fp_count_OOB = aggregated_oob_df['MajorityClass'].value_counts().get('FalsePositive', 0)
fn_count_OOB = aggregated_oob_df['MajorityClass'].value_counts().get('FalseNegative', 0)


# Create a confusion matrix DataFrame
confusion_matrix_local_OOB = pd.DataFrame({
    'Predicted_Positive': [tp_count_OOB, fp_count_OOB],
    'Predicted_Negative': [fn_count_OOB, tn_count_OOB]
}, index=['Actual_Positive', 'Actual_Negative'])


# Print or display the confusion matrix
print('Confusion Matrix for OOB')
print(confusion_matrix_local_OOB)

TP_L_OOB = tp_count_OOB
TN_L_OOB = tn_count_OOB
FP_L_OOB = fp_count_OOB
FN_L_OOB = fn_count_OOB

# Calculate performance metrics
accuracy_local_OOB = (TP_L_OOB + TN_L_OOB) / (TP_L_OOB + TN_L_OOB + FP_L_OOB + FN_L_OOB)
precision_local_OOB = TP_L_OOB / (TP_L_OOB + FP_L_OOB)
recall_local_OOB = TP_L_OOB / (TP_L_OOB + FN_L_OOB)
f1_local_OOB = 2 * precision_local_OOB * recall_local_OOB / (precision_local_OOB + recall_local_OOB)


# Print or display the calculated metrics
print("OOB Accuracy: {:.2f}%".format(accuracy_local_OOB*100))
print("OOB Precision: {:.2f}".format(precision_local_OOB*100))
print("OOB Recall: {:.2f}".format(recall_local_OOB*100))
print("OOB F1 Score: {:.2f}".format(f1_local_OOB*100))

print('############ Local model Metrics PREDICTION 2nd method #############')

actual_values_2nd = prediction_row_2nd_df['y_test_l'].astype(int)
predicted_values_2nd = prediction_row_2nd_df['y_pred_l'].astype(int)

confusion_matrix_result_2nd = confusion_matrix(actual_values_2nd, predicted_values_2nd)

print('Confusion Matrixof PREDICTION for 2nd method')
print(confusion_matrix_result_2nd)

# Calculate accuracy
accuracy_local_2nd = accuracy_score(actual_values_2nd, predicted_values_2nd)

# Calculate precision
precision_local_2nd = precision_score(actual_values_2nd, predicted_values_2nd)

# Calculate recall
recall_local_2nd = recall_score(actual_values_2nd, predicted_values_2nd)

# Calculate F1 score
f1_local_2nd = f1_score(actual_values_2nd, predicted_values_2nd)

# Print the results
print("Accuracy for the 2nd method of prediction: {:.2f}%".format(accuracy_local_2nd*100))
print("precision for the 2nd method of prediction: {:.2f}%".format(precision_local_2nd*100))
print("Recall for the 2nd method of prediction: {:.2f}%".format(recall_local_2nd*100))
print("F1 Score for the 2nd method of prediction: {:.2f}%".format(f1_local_2nd*100))


print('finish')


