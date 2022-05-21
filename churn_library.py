'''
Module: Data analysis and data mining with machine learning models

Author: Nontawat Pattanajak
Date: 28 April 2022
'''

# Import libraries
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    return pd.read_csv(pth)


def perform_eda(data_df):
    '''
    perform eda on df and save figures to images folder
    input:
            df_data: pandas dataframe

    output:
            None
    '''

    # Explore number of exsiting customer
    data_df['Churn'] = data_df['Attrition_Flag']\
        .apply(lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    data_df['Churn'].hist()
    plt.savefig('./images/eda/hist_churn.jpeg')

    # Explore customer age
    plt.figure(figsize=(20, 10))
    data_df['Customer_Age'].hist()
    plt.savefig('./images/eda/hist_customer_age.jpeg')

    # Explore marital status
    plt.figure(figsize=(20, 10))
    data_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/plot_Marital_Status.jpeg')

    # Distribution of Total Trans CT
    plt.figure(figsize=(20, 10))
    sns.distplot(df['Total_Trans_Ct'])
    plt.savefig('./images/eda/hist_Total_Trans_Ct.jpeg')

    # Heatmap of data features
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap_corr.jpeg')

    return data_df


def encoder_helper(data_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that
                      could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    for title in category_lst:

        # Define empty list
        data_lst = []

        # Process data by title
        data_groups = df.groupby(title).mean()['Churn']

        # Collect value
        for val in df[title]:
            data_lst.append(data_groups.loc[val])

        # Add new colume (name feature) if required by user
        if response:
            data_df[title + '_' + response] = data_lst
        else:
            data_df[title] = data_lst

    return data_df


def perform_feature_engineering(data_df, response):
    '''
    input:
              data_df: pandas dataframe
              response: string of response name [optional argument that
                        could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    cat_columns = ['Gender',
                   'Education_Level',
                   'Marital_Status',
                   'Income_Category',
                   'Card_Category']

    encoder_df = encoder_helper(data_df, cat_columns, response)

    # Define label data
    y = data_df['Churn']

    # Define feature data
    X = pd.DataFrame()

    keep_cols = ['Customer_Age', 'Dependent_count',
                 'Months_on_book', 'Total_Relationship_Count',
                 'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
                 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                 'Total_Trans_Amt', 'Total_Trans_Ct',
                 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_' + response, 'Education_Level_' + response,
                 'Marital_Status_' + response, 'Income_Category_' + response,
                 'Card_Category_' + response]

    X[keep_cols] = encoder_df[keep_cols]

    # Split data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Random Forest Model
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01,
             1.25,
             str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.6,
             str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')

    plt.savefig(fname='./images/results/rf_results.png')

    # Logistic Regression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01,
             1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01,
             0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')

    # Save result
    plt.savefig(fname='./images/results/logistic_results.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save result
    plt.savefig('./images/results/feature_importance.jpeg')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    # Define ML mdoel
    rfc = RandomForestClassifier(random_state=42, verbose=1)
    lrc = LogisticRegression(verbose=1)

    # Define grid search parameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Define gride search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # Train Random Forest model with grid search
    print('Random Forest model is training')
    cv_rfc.fit(X_train, y_train)

    # Train Logistic Regression
    print('Logistic Regression model is training')
    lrc.fit(X_train, y_train)

    # Run prediction of Random Forest model
    print('Random Forest model is running prediction')
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Run prediction of Logistic Regression
    print('Logistic Regression model is running prediction')
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Plot ROC curve for Logistic Regression
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Plot ROC curve for Random Forest
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)

    # Save result
    plt.savefig(fname='./images/results/roc_curve_result.png')

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    # Define output path to record information about feature importance
    output_path = {cv_rfc: './images/results/feature_importance_rfc.jpeg',
                   lrc: './images/results/feature_importance_lrc.jpeg'}

    # Run feature importance
    for model in (cv_rfc, lrc):
        try:
            feature_importance_plot(model, X_train, output_path[model])
        except BaseException:
            print('no feature importance support')


if __name__ == '__main__':

    # Define data source and response (to rename column name)
    PATH = './data/bank_data.csv'
    response = 'Churn'

    # Import data
    df = import_data(PATH)

    # Exploring Data Analysis
    print('Start doing EDA')
    df = perform_eda(df)

    # Split Data
    print('Start doing Feature Engineering')
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, response)

    # Train model
    print('Start training model')
    train_models(X_train, X_test, y_train, y_test)
