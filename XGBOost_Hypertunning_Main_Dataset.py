
import pickle as pk
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy.stats import randint, uniform
import xgboost as xgb



################### Configuration #####################

Preanalysis = False

GlobalModel = True
Classifiction = True


DatasetToAnalyze = 'Main' # Main , Distance, Windows
################### Import the dataset #####################

os.chdir("/cephyr/users/alihaki/Alvis/Desktop/Duststorm/DustStormModeling/For training/")

if DatasetToAnalyze == 'Main':
    dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP___WS'
elif DatasetToAnalyze == 'Distance':
    dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP__Dist_WS'
elif DatasetToAnalyze == 'Windows':
    dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_WMe___WS'

dataset = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))
dataset = dataset.dropna()

dataset.reset_index(drop=True, inplace=True)


if DatasetToAnalyze == 'Main':
    df = dataset.drop(columns=['Year', 'Profile_curvature', 'Plan_curvature','X','Y'])
elif DatasetToAnalyze == 'Distance':
    df = dataset.drop(columns=['Year', 'Profile_curvature', 'Plan_curvature',
                                    'Lakes','soil_type_Silt','X','Y',
                                    'landcover_Cropland','landcover_Natural_vegetation'])
elif DatasetToAnalyze == 'Windows':
    dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_WMe___WS'


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
# Define hyperparameter search space
param_dist = {
    'n_estimators': randint(50, 1500),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 0.5),
    'colsample_bytree': uniform(0.5, 0.5),
    'gamma': uniform(0, 1),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
}

# Initialize XGBoost classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=xgb_classifier,
                                   param_distributions=param_dist,
                                   n_iter=500000,  # Adjust the number of iterations as needed
                                   scoring='accuracy',  # Choose appropriate scoring metric
                                   cv=5,  # Number of cross-validation folds
                                   verbose=0,
                                   random_state=0,
                                   n_jobs=-1)  # Utilize all available CPU cores

# Fit the RandomizedSearchCV object to the training data
random_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

# Get the best model
best_model = random_search.best_estimator_

# Evaluate the best model on the test set
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)