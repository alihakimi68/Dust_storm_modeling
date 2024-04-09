
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
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import random
import seaborn as sns
from sklearn.preprocessing import binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold


################### Configuration #####################

Preanalysis = False

GlobalModel = True
Classifiction = True



################### Import the dataset #####################

os.chdir("/For training/")


dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP__'
# dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP__Dist_WS'


df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))

df = df.dropna()

df.reset_index(drop=True, inplace=True)

coords = df[['X', 'Y']]




######## No Dist
df = df.drop(columns=['Year','Profile_curvature','Plan_curvature','X','Y'])
Numerical_cols = ['Soil_evaporation', 'Precipitation', 'Soil_moisture', 'NDVI',
       'Elevation', 'Aspect', 'Curvature', 'Distance_to_river', 'Slope', 'Wind_Speed']

Categorical_cols = ['Lakes','landcover_Cropland','landcover_Natural_vegetation',
                'soil_type_Clay_Loam', 'soil_type_Loam','soil_type_Loam_Sand',
                'soil_type_Sand', 'soil_type_Sand_Clay_Loam','soil_type_Sand_Loam',
                'soil_type_Silt']

# ######## With Dist
# df = df.drop(columns=['Year','Profile_curvature','Plan_curvature','X','Y','Lakes','landcover_Cropland','landcover_Natural_vegetation','soil_type_Clay_Loam'])
# df = df[['Soil_evaporation', 'Distance_to_lakes', 'Precipitation', 'Soil_moisture', 'NDVI',
#        'Elevation', 'Aspect', 'Curvature', 'Distance_to_river', 'Slope', 'Wind_Speed','dust_storm',
#          'Distance_to_cropland','Distance_to_SparseVegetation','Distance_to_DenseVegetation',
#          'soil_type_Loam','soil_type_Loam_Sand','soil_type_Sand', 'soil_type_Sand_Clay_Loam',
#          'soil_type_Sand_Loam','soil_type_Silt']]
#
#
# Numerical_cols = ['Soil_evaporation','Distance_to_lakes', 'Precipitation', 'Soil_moisture', 'NDVI',
#                 'Elevation', 'Aspect', 'Curvature', 'Distance_to_river', 'Slope', 'Wind_Speed',
#                 'Distance_to_cropland','Distance_to_SparseVegetation','Distance_to_DenseVegetation']
#
# Categorical_cols = ['soil_type_Loam','soil_type_Loam_Sand','soil_type_Sand',
#                  'soil_type_Sand_Clay_Loam','soil_type_Sand_Loam','soil_type_Silt']



if Preanalysis:
    correlation_matrix = df.corr()

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Create a heatmap with values displayed
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

    # Customize the plot
    plt.title('Correlation Matrix Heatmap')
    plt.xticks(rotation=25, ha='right')
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Set style and context for seaborn
    sns.set(style="ticks")

    # Combine the plots for all numerical columns into a single pair plot
    plt.figure(figsize=(15, 10))
    pair_plot = sns.pairplot(df, hue='dust_storm', markers=["o", "s"], vars=Numerical_cols, height=2.5, aspect=1, kind='reg')
    for ax in pair_plot.axes.flat:
        ax.yaxis.label.set_rotation(0)
        ax.yaxis.label.set_ha('right')
    plt.suptitle("Pair Plots for Numerical Columns against dust_storm", y=1.02)
    plt.show()

    # Calculate the number of rows and columns based on the length of Categorical_cols
    n_cols = 2
    n_rows = -(-len(Categorical_cols) // n_cols)  # Ceiling division to ensure enough rows

    # Create subplots with a better layout
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6 * n_rows))

    # Flatten the 2D array of axes to make it easier to index
    axes = axes.flatten()

    # Remove any empty subplots
    for i in range(len(Categorical_cols), n_rows * n_cols):
        fig.delaxes(axes[i])

    # Visualize 'dust_storm' against Categorical_cols in subplots
    for idx, col in enumerate(Categorical_cols):
        sns.countplot(x=col, hue='dust_storm', data=df, ax=axes[idx], palette="husl", edgecolor=".6")
        axes[idx].set_title(f'Dust Storm Distribution for {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Assuming 'dust_storm' is the column you want to exclude
columns_to_scale = [col for col in df.columns if col not in ['dust_storm','Lakes']]

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the specified columns
df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

X = df.drop(['dust_storm'], axis=1)
y = df['dust_storm']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)


# Additional stopping parameters
param_grid = {
    # Existing hyperparameters
    'n_estimators': [i for i in range(40, 200)],
    'max_depth': [i for i in range(7, 13)],
    'max_features': [6, 7, 8, 9, 10, 11, 12],
    'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5, 6],
    'criterion': ['entropy', 'gini'],
    'bootstrap': [True, False],
    'min_weight_fraction_leaf': [i/10 for i in range(0, 6)],
    'max_leaf_nodes': [i for i in range(10, 101)],
    'min_impurity_decrease': [i/100 for i in range(0, 21)],
    'ccp_alpha': [i/1000 for i in range(0, 101)],
    'max_samples': [i/10 for i in range(1, 11)],

    # Additional stopping parameters
    'cv': [4, 5, 6, 10,15],  # Number of folds for cross-validation
    'scoring': ['accuracy', 'precision'],  # Metrics to optimize
}

rf_classifier = RandomForestClassifier()

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

print("Best Hyperparameters:", best_params)

# Get the best model
best_model = GridSearchCV.best_estimator_

# Evaluate the best model on the test set
accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", accuracy)


# Perform 5-fold cross-validation
nsplits = 10
cv = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=0)

# n_estimators = [50, 100, 150, 200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
# n_estimators = list(range(40, 151))
n_estimators = [60, 70, 90, 100,120]

# Initialize lists to store results
Global_RF_C_test_accuracies = []
Global_RF_C_feature_importance_scores = []
Global_RF_C_cross_scores_mean = []

Global_RF_R_test_accuracies = []
Global_RF_R_feature_importance_scores = []
Global_RF_R_cross_scores_mean = []

Global_xgb_C_test_accuracies = []
Global_xgb_C_feature_importance_scores = []
Global_xgb_C_cross_scores_mean = []

Global_xgb_R_test_accuracies = []
Global_xgb_R_feature_importance_scores = []
Global_xgb_R_cross_scores_mean = []

threshold = 0.5  # Adjust the threshold as needed

if GlobalModel and Classifiction:
    for n_estimator in n_estimators:

        # Random Forest Classification
        Global_RF_C = RandomForestClassifier(n_estimators=n_estimator, random_state=0)
        Global_RF_C_cross_scores = cross_val_score(Global_RF_C, X, y, cv=cv, scoring='accuracy')
        Global_RF_C_cross_scores_mean.append(Global_RF_C_cross_scores.mean())
        Global_RF_C.fit(X_train, y_train)
        Global_RF_C_Predict = Global_RF_C.predict(X_test)
        Global_RF_C_test_accuracies.append(Global_RF_C.score(X_test, y_test))
        Global_RF_C_feature_importance_scores.append(Global_RF_C.feature_importances_)

        # Random Forest Regression
        Global_RF_R = RandomForestRegressor(n_estimators=n_estimator, random_state=0)
        Global_RF_R_cross_scores = cross_val_score(Global_RF_R, X, y, cv=cv, scoring='neg_mean_squared_error')
        Global_RF_R_cross_scores_mean.append(Global_RF_R_cross_scores.mean())
        Global_RF_R.fit(X_train, y_train)
        Global_RF_R_Predict = Global_RF_R.predict(X_test)
        # Convert probabilities to binary classification
        Global_RF_R_Predict_Binary = binarize([Global_RF_R_Predict], threshold=threshold)[0]
        accuracy = accuracy_score(y_test, Global_RF_R_Predict_Binary)
        Global_RF_R_test_accuracies.append(accuracy)
        Global_RF_R_feature_importance_scores.append(Global_RF_R.feature_importances_)

        # XGBoost Classification
        Global_xgb_C = xgb.XGBClassifier(n_estimators=n_estimator, random_state=0)
        Global_xgb_C_cross_scores = cross_val_score(Global_xgb_C, X, y, cv=cv, scoring='accuracy')
        Global_xgb_C_cross_scores_mean.append(Global_xgb_C_cross_scores.mean())
        Global_xgb_C.fit(X_train, y_train)
        Global_xgb_C_Predict = Global_xgb_C.predict(X_test)
        Global_xgb_C_test_accuracies.append(Global_xgb_C.score(X_test, y_test))
        Global_xgb_C_feature_importance_scores.append(Global_xgb_C.feature_importances_)

        # XGBoost Regression
        Global_xgb_R = xgb.XGBRegressor(n_estimators=n_estimator, random_state=0)
        Global_xgb_R_cross_scores = cross_val_score(Global_xgb_R, X, y, cv=cv, scoring='neg_mean_squared_error')
        Global_xgb_R_cross_scores_mean.append(Global_xgb_R_cross_scores.mean())
        Global_xgb_R.fit(X_train, y_train)
        Global_xgb_R_Predict = Global_xgb_R.predict(X_test)
        # Convert probabilities to binary classification
        Global_xgb_R_Predict_Binary = binarize([Global_xgb_R_Predict], threshold=threshold)[0]
        accuracy = accuracy_score(y_test, Global_xgb_R_Predict_Binary)
        Global_xgb_R_test_accuracies.append(accuracy)
        Global_xgb_R_feature_importance_scores.append(Global_xgb_R.feature_importances_)

# Random Forest Classification / Find the index of the maximum test accuracy
Global_RF_C_max_accuracy_index = Global_RF_C_test_accuracies.index(max(Global_RF_C_test_accuracies))
Global_RF_C_n_estimator = n_estimators[Global_RF_C_max_accuracy_index]
Global_RF_C_crossvalidtion_accuracy = Global_RF_C_cross_scores_mean[Global_RF_C_max_accuracy_index]
Global_RF_C_feature_importance = Global_RF_C_feature_importance_scores[Global_RF_C_max_accuracy_index]

Global_RF_C_importances = np.array(Global_RF_C_feature_importance_scores)
Global_RF_C_min_importances = Global_RF_C_importances.min(axis=0)
Global_RF_C_max_importances = Global_RF_C_importances.max(axis=0)
Global_RF_C_mean_importances = Global_RF_C_importances.mean(axis=0)
# Plotting for Random Forest Classification
plt.figure(figsize=(10, 6))
plt.plot(Global_RF_C_min_importances, label='Min Importance',color='blue')
plt.plot(Global_RF_C_max_importances, label='Max Importance',color='red')
plt.plot(Global_RF_C_mean_importances, label='Mean Importance', linestyle='dashed',color='grey')
plt.plot(Global_RF_C_feature_importance, label='Highest accuracy Importance',color='green')
plt.xticks(np.arange(len(X.columns)), X.columns,rotation=20, ha='right')
plt.title('Random Forest Classification Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.legend()
plt.show()

print(f'For Random Forest Classification with {Global_RF_C_n_estimator} estimators the accuracy is {max(Global_RF_C_test_accuracies)}'
      f',mean accuracy for {nsplits}_folds is {Global_RF_C_crossvalidtion_accuracy}')


# Random Forest Regression / Find the index of the maximum test accuracy
Global_RF_R_max_accuracy_index = Global_RF_R_test_accuracies.index(max(Global_RF_R_test_accuracies))
Global_RF_R_n_estimator = n_estimators[Global_RF_R_max_accuracy_index]
Global_RF_R_crossvalidtion_accuracy = Global_RF_R_cross_scores_mean[Global_RF_R_max_accuracy_index]
Global_RF_R_feature_importance = Global_RF_R_feature_importance_scores[Global_RF_R_max_accuracy_index]

Global_RF_R_importances = np.array(Global_RF_R_feature_importance_scores)
Global_RF_R_min_importances = Global_RF_R_importances.min(axis=0)
Global_RF_R_max_importances = Global_RF_R_importances.max(axis=0)
Global_RF_R_mean_importances = Global_RF_R_importances.mean(axis=0)
# Plotting for Random Forest Regression
plt.figure(figsize=(10, 6))
plt.plot(Global_RF_R_min_importances, label='Min Importance',color='blue')
plt.plot(Global_RF_R_max_importances, label='Max Importance',color='red')
plt.plot(Global_RF_R_mean_importances, label='Mean Importance', linestyle='dashed',color='grey')
plt.plot(Global_RF_R_feature_importance, label='Highest accuracy Importance',color='green')
plt.xticks(np.arange(len(X.columns)), X.columns,rotation=20, ha='right')
plt.title('Random Forest Regression Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.legend()
plt.show()

print(f'For Random Forest Regression with {Global_RF_R_n_estimator} estimators the accuracy is {max(Global_RF_R_test_accuracies)}'
    f',mean accuracy for {nsplits}_folds is {Global_RF_R_crossvalidtion_accuracy}')

# XGBoost Classification / Find the index of the maximum test accuracy
Global_xgb_C_max_accuracy_index = Global_xgb_C_test_accuracies.index(max(Global_xgb_C_test_accuracies))
Global_xgb_C_n_estimator = n_estimators[Global_xgb_C_max_accuracy_index]
Global_xgb_C_crossvalidtion_accuracy = Global_xgb_C_cross_scores_mean[Global_xgb_C_max_accuracy_index]
Global_xgb_C_feature_importance = Global_xgb_C_feature_importance_scores[Global_xgb_C_max_accuracy_index]

Global_xgb_C_importances = np.array(Global_xgb_C_feature_importance_scores)
Global_xgb_C_min_importances = Global_xgb_C_importances.min(axis=0)
Global_xgb_C_max_importances = Global_xgb_C_importances.max(axis=0)
Global_xgb_C_mean_importances = Global_xgb_C_importances.mean(axis=0)
# Plotting for XGBoost Classification
plt.figure(figsize=(10, 6))
plt.plot(Global_xgb_C_min_importances, label='Min Importance',color='blue')
plt.plot(Global_xgb_C_max_importances, label='Max Importance',color='red')
plt.plot(Global_xgb_C_mean_importances, label='Mean Importance', linestyle='dashed',color='grey')
plt.plot(Global_xgb_C_feature_importance, label='Highest accuracy Importance',color='green')
plt.xticks(np.arange(len(X.columns)), X.columns,rotation=20, ha='right')
plt.title('Random XGBoost Classification Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.legend()
plt.show()

print(f'For XGBoost Classification with {Global_xgb_C_n_estimator} estimators the accuracy is {max(Global_xgb_C_test_accuracies)}'
    f',mean accuracy for {nsplits}_folds is {Global_xgb_C_crossvalidtion_accuracy}')


# XGBoost Regression / Find the index of the maximum test accuracy
Global_xgb_R_max_accuracy_index = Global_xgb_R_test_accuracies.index(max(Global_xgb_R_test_accuracies))
Global_xgb_R_n_estimator = n_estimators[Global_xgb_R_max_accuracy_index]
Global_xgb_R_crossvalidtion_accuracy = Global_xgb_R_cross_scores_mean[Global_xgb_R_max_accuracy_index]
Global_xgb_R_feature_importance = Global_xgb_R_feature_importance_scores[Global_xgb_R_max_accuracy_index]

Global_xgb_R_importances = np.array(Global_xgb_R_feature_importance_scores)
Global_xgb_R_min_importances = Global_xgb_R_importances.min(axis=0)
Global_xgb_R_max_importances = Global_xgb_R_importances.max(axis=0)
Global_xgb_R_mean_importances = Global_xgb_R_importances.mean(axis=0)
# Plotting for XGBoost Regression
plt.figure(figsize=(10, 6))
plt.plot(Global_xgb_R_min_importances, label='Min Importance',color='blue')
plt.plot(Global_xgb_R_max_importances, label='Max Importance',color='red')
plt.plot(Global_xgb_R_mean_importances, label='Mean Importance', linestyle='dashed',color='grey')
plt.plot(Global_xgb_R_feature_importance, label='Highest accuracy Importance',color='green')
plt.xticks(np.arange(len(X.columns)), X.columns,rotation=20, ha='right')
plt.title('Random XGBoost Regression Feature Importances')
plt.xlabel('Feature Index')
plt.ylabel('Importance Score')
plt.legend()
plt.show()
print(f'For XGBoost Regression with {Global_xgb_R_n_estimator} estimators the accuracy is {max(Global_xgb_R_test_accuracies)}'
    f',mean accuracy for {nsplits}_folds is {Global_xgb_R_crossvalidtion_accuracy}')


# Plot the results
plt.figure(figsize=(10, 6))

plt.plot(n_estimators, Global_RF_C_test_accuracies, marker='o',
         label='Random Forest Classification', color='blue')
plt.plot(n_estimators, Global_RF_R_test_accuracies, marker='o',
         label='Random Forest Regression', color='green')
plt.plot(n_estimators, Global_xgb_C_test_accuracies, marker='o',
         label='XGBoost Classification', color='red')
plt.plot(n_estimators, Global_xgb_R_test_accuracies, marker='o',
         label='XGBoost Regression', color='purple')
plt.plot(n_estimators, Global_RF_C_cross_scores_mean, marker='.',
         label='Random Forest Classification 5 fold CV', color='blue', alpha=0.5)
# plt.plot(n_estimators, Global_RF_R_cross_scores_mean, marker='.',
#          label='Random Forest Regression 5 fold CV', color='green', alpha=0.5)
plt.plot(n_estimators, Global_xgb_C_cross_scores_mean, marker='.',
         label='XGBoost Classification 5 fold CV', color='red', alpha=0.5)
# plt.plot(n_estimators, Global_xgb_R_cross_scores_mean, marker='.',
#          label='XGBoost Regression 5 fold CV', color='purple', alpha=0.5)




plt.title('Test Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


print('finish')
