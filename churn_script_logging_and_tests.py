'''
Module: script to test functions in churn_library.py

Author: Nontawat Pattanajak
Date: 28 April 2022
'''

import os
import logging
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing output of import_data(): SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    
    dataFrame = cl.import_data("./data/bank_data.csv")
    
    # To test perform_eda() function in general
    try:
        perform_eda(dataFrame)
        logging.info("Testing perform_eda(): SUCCESS")
    except Exception as error:
        print("FAILED: Error was found. - ", error)
        logging.error("Testing perform_eda(): FAILED")
    
    # To test that output is not empty
    try:
        assert os.path.isfile("./images/eda/hist_churn.jpeg") is True
        logging.info("Testing output data in perform_eda(): SUCCESS")
    except BaseException as err:
        logging.error("Testing output data in perform_eda(): FAILED")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    
    dataFrame = cl.import_data("./data/bank_data.csv")

    # To test encoder_helper() function in general
    try:
        data_df = encoder_helper(dataFrame, [], 'Churn')
        logging.info("Testing encoder_helper(): SUCCESS")
    except Exception as error:
        print("FAILED: Error was found. - ", error)
        logging.error("Testing encoder_helper(): FAILED")
    
    # To test that a dataframe is not empty
    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
        logging.info("Testing input data in encoder_helper(): SUCCESS")
    except BaseException:
        logging.error("Testing input data in encoder_helper(): FAILED")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    
    dataFrame = cl.import_data("./data/bank_data.csv")
    
    # To test perform_feature_engineering() function in general    
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(dataFrame)
        logging.info("Testing perform_feature_engineering(): SUCCESS")
    except Exception as error:
        print("FAILED: Error was found. - ", error)
        logging.error("Testing perform_feature_engineering(): FAILED")
    
    # To test that output data is not empty
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_train.shape[1] > 0
        assert y_test.shape[0] > 0
        assert y_test.shape[1] > 0
        logging.info("Testing output data of perform_feature_engineering(): SUCCESS")
    except BaseException:
        logging.error("Testing output data of perform_feature_engineering(): FAILED")


def test_train_models(train_models):
    '''
    test train_models
    '''
    
    #dataFrame = cl.import_data("./data/bank_data.csv")
    #X_train, X_test, y_train, y_test = cl.perform_feature_engineering(dataFrame, '-')
    
    
    # To test that input data is not empty
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_train.shape[1] > 0
        assert y_test.shape[0] > 0
        assert y_test.shape[1] > 0
        logging.info("Testing input data of train_models(): SUCCESS")
    except BaseException:
        logging.error("Testing input data of train_models(): FAILED")
    
    # To test train_models() function in general 
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models(): SUCCESS")
    except Exception as error:
        print("FAILED: Error was found. - ", error)
        logging.error("Testing train_models(): FAILED")


if __name__ == "__main__":
    '''
    Main code to excute functions
    '''

    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)
