# Dust_storm_modeling
Include spatial neighborhood to improve the prediction accuracy of a predictive model using XGBoost algorithm.

To create the environment and install the required packages:

## For anaconda:
  - open the anaconda promt
  - cd to the location of the project
  - run the Conda_Env_setup.bat file
  - wait until the installation finish
  - check the enviroment installation by typing:
      - conda activate Dstorm_311
      - conda list -n Dstorm_311

## required packages:

- The code runs under Python version 3.11 with below packages
  - geopandas==0.14.1
  - rasterio==1.3.9
  - matplotlib==3.8.1
  - scikit-learn==1.3.2
  - numpy==1.26.2
  - pickles==0.1.1
  - pandas==2.1.3
  - xgboost==2.0.2
  - scipy==1.11.4

## Python Files Decription


- Combo_randomForest.py / Combo_XGBoost.py
  - Random Search for tuning Spatial parameters

- Feature_Selection_RF.py / Feature_Selection_XGBoost.py
  - Dimentionality reduction and Feature Selection

- GWML_ANN.py / GWML_RandomForest.py / GWML_XGBoost.py
  - Binary Classification GWML

- GWML_RandomForest_Australia.py / GWML_XGBoost_Australia.py
  - Regression GWML

- RandomFOrest_Hypertunning_Main_Dataset.py / XGBOost_Hypertunning_Main_Dataset.py
  - Random Search for huper paramter tuning (NAISS)

- Test_bandwidth_GWML_RandomForest.py / Test_bandwidth_GWML_XGBoost.py
  - Bandwidth Exploration

- Test_HyperParameters_GWML_XGBoost.py / Test_HyperParameters_GWML_XGBoost.py
  - Hyper Parameter Exploration  (NAISS)

- Test_Test_GWML_RandomForest.py / Test_Test_GWML_XGBoost.py
  - Data size Exploration  (NAISS)