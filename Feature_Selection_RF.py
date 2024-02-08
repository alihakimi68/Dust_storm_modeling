
import os
import pickle as pk
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, RFE, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import matplotlib.pyplot as plt
import numpy as np

os.chdir("D:/University/DustStorming/ToAli/DustStormModeling/For training/")

# For df_dustsources_WS0_X_0_PN20_SP_
# dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP_'
# FeatureCount = 15

# For df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod
dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod'
FeatureCount = 22
Estimator = 400
# For df_dustsources_WS7_X_7_PN20_SP_WMe_
# dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_WMe_'
# FeatureCount = 15
df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))

# drop original categorical columns
df = df.drop(columns=['X', 'Y','Year', 'landcover', 'soil_type'])


# df_copy = df.drop(columns=['Lakes','Bare_Soil','Cropland','Natural_vegetation','Clay_Loam','Loam','Loam_Sand','Sand','Sand_Clay_Loam','Sand_Loam'])
# # Create a pairplot with linear regression lines
# sns.pairplot(df_copy, kind='reg', markers='.', height=2.5)
# plt.show()

X = df.drop(['dust_storm'], axis=1)
sc = MinMaxScaler()
X_St = sc.fit_transform(X)
X_St = pd.DataFrame(X_St, columns=X.columns)
y = df['dust_storm']
X_train, X_test, y_train, y_test = train_test_split(X_St,y,test_size=0.2,random_state=0,stratify=y)

#####################################################
############## Select K Best Features ###############
#####################################################
print('Best Feature names based on the Select K Best Features anaylsis')

print(X_St.shape)
SelBestK = SelectKBest(chi2, k=FeatureCount)
X_new = SelBestK.fit_transform(X_St, y)
print(X_new.shape)
Feature_names = SelBestK.get_feature_names_out(input_features=None)
print(Feature_names)

print('##############################################')
#####################################################
################ CHi2 Best Features #################
#####################################################
print('CHi2 Best Feature analysis')

chi2_stats, p_values = chi2(X_St, y)
# Create a DataFrame to display the results
results_df = pd.DataFrame({'Feature': X.columns, 'Chi2_Stat': chi2_stats, 'P_Value': p_values})

# Display the results sorted by p-values (lower p-value indicates higher significance)
results_df.sort_values(by='P_Value', ascending=True, inplace=True)

# Print or inspect the results
print(results_df)
print('##############################################')

#####################################################
######### L1 Regularization Best Features ###########
#####################################################
# print('Best Feature names based on the RandomForestClassifier analysis')
#
# # Initialize the Random Forest classifier
# rf = RandomForestClassifier(n_estimators=Estimator, random_state=42)
#
# # Fit the Random Forest model
# rf.fit(X_train, y_train)
#
# print('Training accuracy: ', rf.score(X_train, y_train))
# print('Test accuracy: ', rf.score(X_test, y_test))
# y_pred = rf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# f1 = f1_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# auc = roc_auc_score(y_test, y_pred)
#
# # Print the metrics
# print("Accuracy: {:.2f}%".format(accuracy * 100))
# print("Precision: {:.2f}%".format(precision * 100))
# print("Recall: {:.2f}%".format(recall * 100))
# print("F1-score: {:.2f}%".format(f1 * 100))
# print('Confusion matrix:\n True negative: %s \
#           \n False positive: %s \n False negative: %s \n True positive: %s'
#       % (conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]))
# print('AUC: {:.2f}%'.format(auc * 100))
#
# # No coefficients in Random Forest, but you can check feature importances
# feature_importances = rf.feature_importances_
# print('Feature importances: ', feature_importances)
#
# # Set a range of n_estimators values to explore (you can modify the parameters as needed)
# n_estimators_values = [50, 100, 150, 200,300,400,500,600,700,800,900,1000,1100,1200]
#
# # Initialize lists to store results
# train_accuracies = []
# test_accuracies = []
# feature_importance_scores = []
#
# # Loop through different n_estimators values
# for n_estimators in n_estimators_values:
#     rf2 = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#     rf2.fit(X_train, y_train)
#
#     # Store accuracy scores
#     train_accuracies.append(rf2.score(X_train, y_train))
#     test_accuracies.append(rf2.score(X_test, y_test))
#
#     # Store the feature importances
#     feature_importance_scores.append(rf2.feature_importances_)
#
# # Plot the results
# plt.figure(figsize=(10, 6))
#
# # Plot Training Accuracy
# plt.subplot(2, 1, 1)
# plt.plot(n_estimators_values, train_accuracies, marker='o')
# plt.title('Training Accuracy vs. Number of Trees')
# plt.xlabel('Number of Trees (n_estimators)')
# plt.ylabel('Training Accuracy')
#
# # Plot Test Accuracy
# plt.subplot(2, 1, 2)
# plt.plot(n_estimators_values, test_accuracies, marker='o')
# plt.title('Test Accuracy vs. Number of Trees')
# plt.xlabel('Number of Trees (n_estimators)')
# plt.ylabel('Test Accuracy')
#
# plt.tight_layout()
# plt.show()
#
# # Find the index of the maximum test accuracy
# max_test_accuracy_index = test_accuracies.index(max(test_accuracies))
#
# # Use the index to get the corresponding feature importances
# selected_feature_importances = feature_importance_scores[max_test_accuracy_index]
#
# # Get the column names of X_train
# column_names = X_train.columns.tolist()
#
# # Sort feature importances and column names based on feature importances
# sorted_indices = np.argsort(feature_importance_scores[-1])[::-1]  # Sort in descending order
# sorted_feature_importances = np.array(feature_importance_scores[max_test_accuracy_index])[sorted_indices]
# sorted_column_names = np.array(column_names)[sorted_indices]
#
# # Plot the sorted feature importances
# plt.figure(figsize=(10, 4))
# plt.bar(range(X_train.shape[1]), sorted_feature_importances)
# plt.title('Feature Importances')
# plt.xlabel('Feature Index')
# plt.ylabel('Importance Score')
#
# # Set x-axis labels to the sorted column names
# plt.xticks(range(X_train.shape[1]), sorted_column_names, rotation=45, ha='right')
#
# plt.show()
#
# print('##############################################')

#####################################################
############# Recursive Best Features ###############
#####################################################
print('Best Feature names based on Recursive Feature Selection')
n_estimators_values = [50, 100, 150, 200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
rf_classifier = RandomForestClassifier(n_estimators=Estimator, random_state=42)  # You can adjust parameters as needed
rfe = RFE(estimator=rf_classifier, n_features_to_select=FeatureCount, step=1)
rfe.fit(X_train,y_train)

X_train_sub = rfe.transform(X_train)
X_test_sub = rfe.transform(X_test)

# Initialize lists to store results
train_accuracies = []
test_accuracies = []
feature_importance_scores = []

# Loop through different n_estimators values
for n_estimators in n_estimators_values:
    rf3 = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf3.fit(X_train_sub, y_train)

    # Store accuracy scores
    train_accuracies.append(rf3.score(X_train_sub, y_train))
    test_accuracies.append(rf3.score(X_test_sub, y_test))

    # Store the feature importances
    feature_importance_scores.append(rf3.feature_importances_)

# Plot the results
plt.figure(figsize=(10, 6))

# Plot Training Accuracy
plt.subplot(2, 1, 1)
plt.plot(n_estimators_values, train_accuracies, marker='o')
plt.title('Training Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Training Accuracy')

# Plot Test Accuracy
plt.subplot(2, 1, 2)
plt.plot(n_estimators_values, test_accuracies, marker='o')
plt.title('Test Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Test Accuracy')

plt.tight_layout()
plt.show()

# Find the index of the maximum test accuracy
max_test_accuracy_index = test_accuracies.index(max(test_accuracies))

New_estimator = n_estimators_values[max_test_accuracy_index]
print(f'The best estimator for {FeatureCount} features is {New_estimator} ')
rf4 = RandomForestClassifier(n_estimators=New_estimator, random_state=42)
rf4.fit(X_train_sub, y_train)

# Get the feature importances
feature_importances_rf4 = rf4.feature_importances_

selected_feature_indices = rfe.get_support()

# Get the column names of X_train
column_names_X_train = X_train.columns.tolist()

# Use the selected_feature_indices to get the corresponding column names
column_names = [column_names_X_train[i] for i in range(len(column_names_X_train)) if selected_feature_indices[i]]


# Sort feature importances and column names based on feature importances
sorted_indices_rf4 = np.argsort(feature_importances_rf4)[::-1]  # Sort in descending order
sorted_feature_importances_rf4 = feature_importances_rf4[sorted_indices_rf4]
sorted_column_names_rf4 = np.array(column_names)[sorted_indices_rf4]

# Plot the sorted feature importances for rf4
plt.figure(figsize=(10, 4))
plt.bar(range(X_train_sub.shape[1]), sorted_feature_importances_rf4)
plt.title('Feature Importances (rf4)')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')

# Set x-axis labels to the sorted column names
plt.xticks(range(X_train_sub.shape[1]), sorted_column_names_rf4, rotation=45, ha='right')

plt.show()

print(rfe.support_)

print(df.columns[1:][rfe.support_])

y_pred = rf4.predict(X_test_sub)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
print('Metrics After feature selection')
# Print the metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1-score: {:.2f}%".format(f1 * 100))
print('Confusion matrix:\n True negative: %s \
          \n False positive: %s \n False negative: %s \n True positive: %s'
      % (conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]))
print('AUC: {:.2f}%'.format(auc * 100))

print('##############################################')

#####################################################
############# Sequential Best Features ##############
#####################################################

print('Best Feature names based on Sequential Feature Selection')

rf4.fit(X_train, y_train)

# Predictions on training and test sets
train_predictions = rf4.predict(X_train)
test_predictions = rf4.predict(X_test)

# Calculate and print accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print('Training accuracy: ', train_accuracy)
print('Test accuracy: ', test_accuracy)

sfs = SequentialFeatureSelector(rf4,
                                n_features_to_select=FeatureCount,
                                direction='backward',
                                scoring='accuracy',
                                n_jobs=-1,
                                cv = 5)

sfs = sfs.fit(X_train,y_train)

# Transform the data
X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

# Retrain the Logistic Regression model on the selected features
rf4.fit(X_train_sfs, y_train)

# Predictions on training and test sets
train_predictions = rf4.predict(X_train_sfs)
test_predictions = rf4.predict(X_test_sfs)

# Calculate and print accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print('Training accuracy: ', train_accuracy)
print('Test accuracy: ', test_accuracy)

print(np.arange(X_St.shape[1])[sfs.support_])
print(df.columns[1:][sfs.support_])



