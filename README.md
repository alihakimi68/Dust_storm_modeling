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

## After installation run the xboost_training.py file

-  set the project directory in os.chdir
-  set the required variavles:
    - CreateDataSet = False  # True for creating a dataset from For training folder
    - window_size = 3  # 3,5,7 ... for picking window size to search neighbor pixels of dust source
    - FindBestParam = False  # True for finding the best hyperparameters
    - year_list = list(range(2001, 2021))  # temporal duration to study 2021 is not included
    - CalculateSeasons = False  # True divides data in to 4 periods :
      - First Period is Dry from 2000:2004
      - Second Period is Wet from 2005:2007
      - Third Period is Dry from 2008:2012
      - Fourth Period is Wet from 2012:2020
    - Select which type of statistical parameters should be calculated as input feature

      numerical = {'Mean': False,
             'WMean': True,
             'Variance': False,
             'Covariance': False,
             'Median': True}

      categorical = {'Entropy': True,
               'Mode': False}

## After modeling is finished
- Results will be saved in below format in the "For training" folder
  - If the CreateDataSet = True
    - A dataset with the below naming pattren as *.pickle will be saved
  - If the CalculateSeasons = True
    - For every season a dataset with the below naming pattren as *.pickle will be saved (PN number)
  - If FindBestParam = True
    - The program does through Random search and grid search to hyper tune the parameters
    - It is a time consuming task and can vary from 5 hours to 48 hours depend on the settings and param range
- The name of he file will be shown in the Run console just copy the name search it in the folder
- The naming will be like:
  - WS = windows size 0,3,5,7,9,...
  - PN = period number (20 for all years,4: First Period is Dry from 2000:2004, 3:Second Period is Wet from 2005:2007, 5: Third Period is Dry from 2008:2012, 8:Fourth Period is Wet from 2012:2020 )
  - SP = Statistical Parameters : AVR:Average, VAR: Variance, MED: Median, COV: Covariance, ENT: Entropy, MOD: Mode
- The results will be shown and saved as figure and text
  - Figure shows the feature importance
  - Text shows the accuracy, F1 score, Recall, precision, confiusion matrix, cross validation results
