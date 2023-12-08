@echo off

REM Create a conda environment named Dstorm_311 and install required packages
call conda create --name Dstorm_311 python=3.11 -y
call conda activate Dstorm_311

REM Install required packages
pip install geopandas==0.14.1
pip install rasterio==1.3.9
pip install matplotlib==3.8.1
pip install scikit-learn==1.3.2
pip install numpy==1.26.2
pip install pickles==0.1.1
pip install pandas==2.1.3
pip install xgboost==2.0.2
pip install scipy==1.11.4

REM Deactivate the conda environment
Rem call conda deactivate
