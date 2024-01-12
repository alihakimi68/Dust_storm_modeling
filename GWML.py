import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import pdist, squareform
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import random

Model = 'RF'

data_folder = "D:/University/DustStorming/ToAli/Geographically_weighted_random_forest/"
df = pd.read_csv(data_folder + "df_dustsources_WS0_X_0_PN20_SP_.csv")

# Check and remove NA values
df = df.dropna()

# Columns to convert to factors
columns_to_convert = [2, 13] + list(range(17, 26))
# print(df.columns)
# df.iloc[:, columns_to_convert] = df.iloc[:, columns_to_convert].astype('category')

# Drop 'Year' column
df = df.drop(columns=['Year'])

train_valid_df = df.copy()
coords = train_valid_df[['Y', 'X']]

# Drop the specified coordinates from the train data frame
train_valid_df = train_valid_df.drop(columns=['X', 'Y'])

# Use train_test_split to create indices for random sampling
X = train_valid_df.drop(['dust_storm'], axis=1)
y = train_valid_df['dust_storm']
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2,
                                                                  random_state=42)

# Make a copy of the dataset for the global model
dframe_full = train_valid_df.copy()

# Count the number of observations in the data
obs = len(dframe_full)
params = {}
if Model == 'RF':
    params['n_estimators'] = 890
    params['max_depth'] = 8
    params['max_features'] = 8
    params['random_state'] = 42

    Gl_Model = RandomForestClassifier(**params)
else:
    params['objective'] = 'binary:logistic'
    params['num_class'] = 1
    params['eval_metric'] = 'auc'
    params['learning_rate'] = 0.01
    params['max_depth'] = 7
    params['min_child_weight'] = 2
    params['reg_alpha'] = 0.8999999999999999
    params['reg_lambda'] = 0.7999999999999999
    params['subsample'] = 0.6
    params['gamma'] = 0.1
    params['num_parallel_tree'] = 2

    Gl_Model = xgb.XGBClassifier(**params)
# params['min_samples_split'] = 2



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

feature_importances = Gl_Model.feature_importances_
# Set the importance threshold
importance_threshold = 0

# Identify columns with low importance
low_importance_columns = X_train.columns[feature_importances < importance_threshold]
dframe = dframe_full.drop(low_importance_columns, axis=1)

# Calculate pairwise distances
distance_array = pdist(coords)

# Convert the pairwise distances to a square matrix
Dij = squareform(distance_array)

# Function to calculate geodetic distance between two points
# def calculate_geodetic_distance(point1, point2):
#     return geodesic(point1, point2).kilometers

# Calculate pairwise geodetic distances
# geodetic_distances = []
# for i in range(len(coords)):
#     for j in range(i + 1, len(coords)):
#         point1 = (coords['X'][i], coords['Y'][i])
#         point2 = (coords['X'][j], coords['Y'][j])
#         distance = calculate_geodetic_distance(point1, point2)
#         geodetic_distances.append(distance)
#
# # Convert the pairwise geodetic distances to a square matrix
# geodetic_distance_matrix = pd.DataFrame(squareform(geodetic_distances), index=coords.index, columns=coords.index)


kernel = 'adaptive'
bw = 140
if kernel == 'adaptive':
    Ne = bw
    print(f"Kernel: Adaptive\nNeighbours: {Ne}")
elif kernel == 'fixed':
    print(f"Kernel: Fixed\nBandwidth: {bw}")

prediction_df = pd.DataFrame(columns=['PointID', 'y_test_l', 'y_pred_l', 'TruePositive',
                                      'TrueNegative', 'FalsePositive', 'FalseNegative'])

Validation_OOB_df = pd.DataFrame(columns=['PointID', 'y_OOB', 'y_validation_OOB', 'TruePositive',
                                      'TrueNegative', 'FalsePositive', 'FalseNegative'])


results_data = []

def bootstrapWeighted(X_train, y_train, case_weights):
    # Draw samples with replacement as in-bag and mark as not OOB
    sample_indices = np.random.choice(len(X_train), len(X_train), p=case_weights / np.sum(case_weights))

    # Apply weights to training samples
    X_train_weighted = X_train.iloc[sample_indices]
    y_train_weighted = y_train.iloc[sample_indices]

    # Mimicking the inbag_counts behavior using NumPy
    _, inbag_counts = np.unique(sample_indices, return_counts=True)

    # Find the out-of-bag indices
    oob_indices = np.setdiff1d(np.arange(len(X_train)), sample_indices)

    # Save out-of-bag samples
    X_OOB = X_train.iloc[oob_indices]
    y_OOB = y_train.iloc[oob_indices]

    return X_train_weighted, y_train_weighted, inbag_counts, X_OOB, y_OOB


for m in range(0,50):
    # Get the data
    DNeighbour = Dij[:, m]

    # Add 'pointID' column to 'dframe'
    dframe['pointID'] = range(0, len(dframe))

    # Create a new DataFrame 'DataSet' with 'dframe' and 'DNeighbour'
    DataSet = pd.DataFrame({'DNeighbour': DNeighbour})
    DataSet = pd.concat([dframe, DataSet], axis=1)

    # Sort by distance
    DataSetSorted = DataSet.sort_values(by='DNeighbour')
    if kernel == 'adaptive':

        cc = 1
        # Keep Nearest Neighbours
        SubSet = DataSetSorted.iloc[:Ne, :]

        # Make sure there is at least one type of both labels in the subset
        while len(SubSet['dust_storm'].unique()) < 2 or any(SubSet['dust_storm'].value_counts() < 3):
            SubSet = DataSetSorted.iloc[:Ne + cc, :]
            cc += 1

        Kernel_H = SubSet['DNeighbour'].max()
    elif kernel == 'fixed':
        SubSet = DataSetSorted[DataSetSorted['DNeighbour'] <= bw]
        Kernel_H = bw

    # Use train_test_split to create indices for random sampling
    Xl = SubSet.drop(['dust_storm'], axis=1)
    yl = SubSet['dust_storm']
    X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(Xl, yl, test_size=0.2,
                                                        random_state=42)

    while len(y_train_l.unique()) < 2 or len(y_test_l.unique()) < 2 or any(y_train_l.value_counts() < 1) or any(
            y_test_l.value_counts() < 1):
        random_integer = random.randint(15, 45)
        X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(Xl, yl, test_size=0.2,
                                                                    random_state=random_integer)

    # Bi-square weights
    Wts_train = (1 - (X_train_l['DNeighbour'] / Kernel_H) ** 2) ** 2
    # Gaussian weights
    # Wts_train = np.exp(-(X_train_l['DNeighbour']**2) / (2 * Kernel_H**2))

    # Use bootstrapWeighted to get weighted samples
    X_train_l_weighted, y_train_l_weighted, inbag_counts, X_OOB, y_OOB = bootstrapWeighted(X_train_l,
                                                                                               y_train_l, Wts_train)

    X_train_l_noPID = X_train_l_weighted.drop(['pointID'], axis=1)
    X_test_l_noPID = X_test_l.drop(['pointID'], axis=1)
    X_OOB_noPID = X_OOB.drop(['pointID'], axis=1)


    # X_train_l_noPID = X_train_l_noPID.drop(['DNeighbour'], axis=1)
    # X_test_l_noPID = X_test_l_noPID.drop(['DNeighbour'], axis=1)

    if Model == 'RF':

        params['class_weight'] = 'balanced'
        params['n_jobs'] = -1
        LO_Model = RandomForestClassifier(**params)

    else:
        # params['class_weight'] = 'balanced'
        params['n_jobs'] = -1
        LO_Model = xgb.XGBClassifier(**params)

    # FIT THE MODEL TO THE TRAINING DATA
    LO_Model.fit(X_train_l_noPID, y_train_l_weighted)

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
    'Actual_Positive': [tp_count, fn_count],
    'Actual_Negative': [fp_count, tn_count]
}, index=['Predicted_Positive', 'Predicted_Negative'])

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
    'Actual_Positive': [tp_count_OOB, fn_count_OOB],
    'Actual_Negative': [fp_count_OOB, tn_count_OOB]
}, index=['Predicted_Positive', 'Predicted_Negative'])

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


print('finish')


