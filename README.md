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
    - CalculateSeasons = True  # divide data in to 4 periods :
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
- Results will be saved in below format
  - WS = windows size 0,3,5,7,9,...
  - PN = period number (20 for all years,4: First Period is Dry from 2000:2004, 3:Second Period is Wet from 2005:2007, 5: Third Period is Dry from 2008:2012, 8:Fourth Period is Wet from 2012:2020 )
  - SP = Statistical Parameters : AVR:Average, VAR: Variance, MED: Median, COV: Covariance, ENT: Entropy, MOD: Mode
- the the result will be shonw in the as figure and text
  -   Figure shows the feature importance
  -   Text shows the accuracy, F1 score, Recall, precision, cross validation results
