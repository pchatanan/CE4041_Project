# Allstate Claims Severity - Machine Learning

 This project is part of **NTU CE4041 Course Project**.

## Developer's guide

 - Raw data is stored at `.\raw data\Kaggle` directory. Training data file is quite large, containing over 500,000 training data pairs. Hence, subset of training data is extracted and stored at `.\raw data\subset`. These files contain training data pairs from 1 to 10,000.

## Get started

 1. Install `virtualenv` package:
    
    > pip install virtualenv

 2. Create virtual environment

    > virtualenv venv

 3. Activate the created virtual environment

     - Using Git Bash (reccommended):
 
    > . venv/Scripts/activate

    - Using Command Prompt:
 
    > venv\Scripts\activate.bat
  
 4. Install requirements:
 
    > pip install -r requirements.txt

 If you are using PyCharm IDE, please configure the IDE's interpreter to use virtual environment created.

## Usefil links
  
  One-Hot-Encoding
  - https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
