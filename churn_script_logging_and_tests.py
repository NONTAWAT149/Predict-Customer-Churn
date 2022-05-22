'''
Module: script to test functions in churn_library.py

Author: Nontawat Pattanajak
Date: 28 April 2022
'''

import os
import logging
import churn_library as cl
#import churn_library_solution as cls

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
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        perform_eda()
        logging.info("Testing perform_eda(): SUCCESS")
    except Exception as error:
        print("FAILED: Error was found. - ", error)
        logging.error("Testing perform_eda(): FAILED")


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''

    # To test that a dataframe is not empty
    try:
        assert data_df.shape[0] > 0
        assert data_df.shape[1] > 0
        logging.info("Testing input data in encoder_helper(): SUCCESS")
    except BaseException:
        logging.error("Testing input data in encoder_helper(): FAILED")

    # To test encoder_helper() function in general
    try:
        encoder_helper()
        logging.info("Testing encoder_helper(): SUCCESS")
    except Exception as error:
        print("FAILED: Error was found. - ", error)
        logging.error("Testing encoder_helper(): FAILED")


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    try:
        perform_feature_engineering()
        logging.info("Testing perform_feature_engineering(): SUCCESS")
    except Exception as error:
        print("FAILED: Error was found. - ", error)
        logging.error("Testing perform_feature_engineering(): FAILED")


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models()
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
