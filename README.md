# Predict Customer Churn

Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
The proposal of this project is to implement good practice to write clean and efficient code for production.

## Files and data description

Data and code structures

.

├── churn_library.py

├── churn_notebook.ipynb

├── churn_script_logging_and_tests.py

├── data

│   └── bank_data.csv

├── Guide.ipynb

├── images

│   ├── eda

│   │   ├── hist_churn.jpeg

│   │   ├── hist_customer_age.jpeg

│   │   ├── hist_Total_Trans_Ct.jpeg

│   │   └── plot_Marital_Status.jpeg

│   └── results

├── logs

│   └── churn_library.log

├── models

│   ├── logistic_model.pkl

│   └── rfc_model.pkl

├── __pycache__

│   └── churn_library.cpython-36.pyc

├── README.md

├── requirements_py3.6.txt

└── requirements_py3.8.txt



#### Code Description:
churn_notebook.ipynb: original code of model experiment
churn_library.py: script to be ready for running in production
churn_script_logging_and_tests.py: script to test functions of churn_library.py

#### Folder Description:
data: dataset
images: store the output from data exploration
logs: record script excution
models: store trained models

## Running Files

Before running scripts;
Install packages in Terminal by using the following command.

> python -m pip install -r requirements_py3.6.txt

These packages will be installed.

* scikit-learn 0.22       
* shap 0.40.0     
* joblib 0.11
* pandas 0.23.3
* numpy 1.19.5 
* matplotlib 2.1.0      
* seaborn 0.11.2
* pylint 2.7.4
* autopep8 1.5.6

When running scripts;

In Terminal, run ipython following by script file. 

(1) To train model:
> ipython churn_library.py

(2) To test code
> ipython churn_script_logging_and_tests.py

User can access log report on logs/churn_library.log



