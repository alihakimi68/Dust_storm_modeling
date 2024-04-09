
import os
import pickle as pk
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2, RFE, SequentialFeatureSelector
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, lasso_path
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

os.chdir("/For training/")

# For df_dustsources_WS0_X_0_PN20_SP_
# dustsourcespickle = 'df_dustsources_WS0_X_0_PN20_SP_'
# FeatureCount = 15

# For df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod
# dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_Var_Med_Ent_Mod'
# FeatureCount = 20
# For df_dustsources_WS7_X_7_PN20_SP_WMe_
dustsourcespickle = 'df_dustsources_WS7_X_7_PN20_SP_WMe_'
FeatureCount = 15
df = pk.load(open(f'{dustsourcespickle}.pickle', 'rb'))

# drop original categorical columns
df = df.drop(columns=['X', 'Y','Year', 'landcover', 'soil_type'])


df_copy = df.drop(columns=['dust_storm','Lakes','Bare_Soil','Cropland','Natural_vegetation','Clay_Loam','Loam','Loam_Sand','Sand','Sand_Clay_Loam','Sand_Loam'])

# Calculate the pairwise correlation
correlation_matrix = df.corr()
# Display the correlation matrix
print(correlation_matrix)

# Plot a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()

# # Create a pairplot with linear regression lines
# g = sns.pairplot(df_copy, kind='reg', markers='.', height=2.5)
# for ax in g.axes.flat:
#     ax.yaxis.label.set_rotation(45)
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
print('Best Feature names based on the Select L1 Regularization analysis')

# the Smaller C the larger the penalty and more 0 in the lr.coef_ and more features to eliminate
lr = LogisticRegression(penalty='l1', C=10, solver='liblinear', max_iter=10000)
lr.fit(X_train,y_train)

print('Training accuracy: ', lr.score(X_train,y_train))
print('Test accuracy: ' , lr.score(X_test,y_test))

Bias = lr.intercept_
print('Bias: ', Bias)

lr.coef_[lr.coef_!=0].shape
print(lr.coef_)

# Set a range of C values to explore
C_values = np.logspace(-3, 1, 20)

# Initialize lists to store results
train_accuracies = []
test_accuracies = []
non_zero_coeffs = []

# Loop through different C values
for C in C_values:
    lr2 = LogisticRegression(penalty='l1', C=C, solver='liblinear', max_iter=10000)
    lr2.fit(X_train, y_train)

    # Store accuracy scores
    train_accuracies.append(lr2.score(X_train, y_train))
    test_accuracies.append(lr2.score(X_test, y_test))

    # Store the number of non-zero coefficients
    non_zero_coeffs.append(np.sum(lr2.coef_ != 0))

# Plot the results
plt.figure(figsize=(10, 6))

# Plot Training Accuracy
plt.subplot(2, 1, 1)
plt.plot(C_values, train_accuracies, marker='o')
plt.xscale('log')
plt.title('Training Accuracy vs. C Value')
plt.xlabel('C Value (Inverse of Regularization Strength)')
plt.ylabel('Training Accuracy')

# Plot Test Accuracy
plt.subplot(2, 1, 2)
plt.plot(C_values, test_accuracies, marker='o')
plt.xscale('log')
plt.title('Test Accuracy vs. C Value')
plt.xlabel('C Value (Inverse of Regularization Strength)')
plt.ylabel('Test Accuracy')

plt.tight_layout()
plt.show()

# Plot the number of non-zero coefficients
plt.figure(figsize=(10, 4))
plt.plot(C_values, non_zero_coeffs, marker='o')
plt.xscale('log')
plt.title('Number of Non-Zero Coefficients vs. C Value')
plt.xlabel('C Value (Inverse of Regularization Strength)')
plt.ylabel('Number of Non-Zero Coefficients')
plt.show()

# Set a range of alpha values to explore
alphas, coefs, _ = lasso_path(X_St, y, alphas=np.logspace(-3, 1, 20))

feature_names = X_St.columns
# Assuming 'num_features' is the number of features
num_features = len(feature_names)

# Create a list of different line styles
linestyles = ['-', '--', '-.', ':'] * (num_features // 4) + ['-'] * (num_features % 4)


# Plot the LASSO path
plt.figure(figsize=(16, 10))

for i in range(coefs.shape[0]):
    plt.plot(C_values, coefs[i, :], label=f'{feature_names[i]}', linestyle=linestyles[i])

plt.xscale('linear')
plt.xlabel('C Value (Inverse of Regularization Strength)')
plt.ylabel('Weight coefficients')
plt.title('LASSO Path')
plt.legend()
plt.show()


print('##############################################')

#####################################################
############# Recursive Best Features ###############
#####################################################
print('Best Feature names based on Recursive Feature Selection')

lr3 = LogisticRegression(solver='liblinear',random_state=123)
rfe = RFE(estimator=lr2, n_features_to_select=FeatureCount, step=1)
rfe.fit(X_train,y_train)

X_train_sub = rfe.transform(X_train)

print(rfe.support_)

print(df.columns[1:][rfe.support_])


print('##############################################')

#####################################################
############# Sequential Best Features ##############
#####################################################

print('Best Feature names based on Sequential Feature Selection')

lr.fit(X_train, y_train)

# Predictions on training and test sets
train_predictions = lr.predict(X_train)
test_predictions = lr.predict(X_test)

# Calculate and print accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print('Training accuracy: ', train_accuracy)
print('Test accuracy: ', test_accuracy)

sfs = SequentialFeatureSelector(lr,
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
lr.fit(X_train_sfs, y_train)

# Predictions on training and test sets
train_predictions = lr.predict(X_train_sfs)
test_predictions = lr.predict(X_test_sfs)

# Calculate and print accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print('Training accuracy: ', train_accuracy)
print('Test accuracy: ', test_accuracy)

print(np.arange(X_St.shape[1])[sfs.support_])
print(df.columns[1:][sfs.support_])



