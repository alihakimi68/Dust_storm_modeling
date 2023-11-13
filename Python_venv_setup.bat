@echo off

REM Create a Python virtual environment named Dstorm_311 with Python 3.11
python -m pip install virtualenv
python -m virtualenv Dstorm_311 --python=3.11

REM Activate the virtual environment
call Dstorm_311\Scripts\activate

REM Install required packages
pip install geopandas==0.14.1
pip install rasterio==1.3.9
pip install matplotlib==3.8.1
pip install scikit-learn==1.3.2
pip install numpy==1.26.2
pip install pickles==0.1.1
pip install pandas==2.1.3
pip install xgboost==2.0.2

REM Deactivate the virtual environment
deactivate