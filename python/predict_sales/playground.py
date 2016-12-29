import pickle

import numpy as np
import pandas as pd

from predict_sales.data import linear_features, xgb_features
from .functions import remove_before_changepoint, log_transform_train, remove_outliers_lm, select_features, \
    allow_modifications, log_revert_predicted, exp_rmspe, predict_elasticnet, xgb_expm1_rmspe, predict_xgboost

import logging

logger = logging.getLogger(__name__)


##
@allow_modifications(True)
def glm_predictions(train, test):
    ##
    logger.info("Dropping store data before changepoint. Initial shape {0}".format(train.shape))
    remove_before_changepoint(train)
    logger.info("Reduced to {0}".format(train.shape))
    logger.info("Date in train: {0}".format('Date' in train.columns))

    ##
    logger.info("Dropping stores not in test set. Initial shape {0}".format(train.shape))
    train.query('Store in {test_set_stores}'
                        .format(test_set_stores=list(test['Store'].unique())), inplace=True)
    logger.info("Reduced to {0}".format(train.shape))
    logger.info("Date in train: {0}".format('Date' in train.columns))

    ##
    logger.debug("Log transform on sales data")
    log_transform_train(train)

    ##
    train = remove_outliers_lm(train)
    logger.info("Removed outliers, reduced shape {0}".format(train.shape))
    logger.info("Date in train: {0}".format('Date' in train.columns))

    ##
    select_features(train, linear_features, inplace=True)
    logger.info("Selected linear features, shape {0}".format(train.shape))
    logger.info("Date in train: {0}".format('Date' in train.columns))

    ##
    logger.info("Test shape {0}".format(test.shape))
    select_features(test, linear_features, inplace=True)
    logger.info("Test, selected linear features, shape {0}".format(test.shape))

    ##
    logger.info("Running elasticnet predictions")
    fitted_elasticnet = predict_elasticnet(train, test, exp_rmspe, steps=15, step_by=3)

    ##
    errors = fitted_elasticnet.errors
    logger.info("Average cv error over all stores: {0}".format(np.mean(errors)))

    preds = fitted_elasticnet.predicted
    logger.info("Predictions for {0} stores".format(len(preds)))

    ##
    for pred in preds:
        log_revert_predicted(pred, test['Open'])

    ##
    # Combine predictions to single dataframe
    preds = pd.concat(preds)
    logger.info("Combined predictions to single dataframe of shape {0}".format(preds.shape))

    with open('glmnet.pkl', mode='wb') as pkl_file:
        pickle.dump(preds, pkl_file)

    return preds


##
xparams = dict(
    booster="gbtree",
    silent=0,

    eta=0.02,
    max_depth=10,
    subsample=0.9,
    colsample_bytree=0.7,

    objective="reg:linear"
)

nrounds = 3000


##
@allow_modifications(False)
def xgb_predictions(train, test):
    ##
    logger.info("Dropping store data before changepoint. Initial shape {0}".format(train.shape))
    train_tr = remove_before_changepoint(train)
    logger.info("Reduced to {0}".format(train_tr.shape))

    ##
    logger.info("Dropping stores not in test set. Initial shape {0}".format(train_tr.shape))
    train_tr = train_tr.query('Store in {test_set_stores}'
                              .format(test_set_stores=list(test['Store'].unique())))
    logger.info("Reduced to {0}".format(train_tr.shape))

    ##
    logger.debug("Log transform on sales data")
    train_tr = log_transform_train(train_tr)

    ##
    train_tr = select_features(train_tr, xgb_features)
    logger.info("Selected xgboost features, shape {0}".format(train_tr.shape))

    ##
    logger.info("Test shape {0}".format(test.shape))
    test_tr = select_features(test, xgb_features)
    logger.info("Test, selected xgboost features, shape {0}".format(test_tr.shape))

    ##
    logger.info("Running xgboost predictions")
    fitted_xgboost = predict_xgboost(train_tr, test_tr, xgb_expm1_rmspe,
                                     params=xparams, nrounds=nrounds)
    # errors = fitted_xgboost.errors

    ##
    # logger.info("Average error over all stores: {0}".format(np.mean(errors)))
    pred = fitted_xgboost.predicted

    log_revert_predicted(pred)

    with open('xgboost.pkl', mode='wb') as pkl_file:
        pickle.dump(pred, pkl_file)

    return pred


##
def mix_models():
    ##
    with open('glmnet.pkl', mode='rb') as pkl_file:
        glm_preds = pickle.load(pkl_file)

    with open('xgboost.pkl', mode='rb') as pkl_file:
        xgb_preds = pickle.load(pkl_file)

    assert glm_preds.shape[0] == xgb_preds.shape[0]

    ##
    joined = pd.merge(glm_preds, xgb_preds, how='inner', on='Id')
    joined['Sales'] = 0.985 * (joined['PredictedSales_x'] + joined['PredictedSales_y']) / 2
    assert joined.shape[0] == glm_preds.shape[0]

    with open('mix.pkl', mode='wb') as pkl_file:
        pickle.dump(joined, pkl_file)

    ##
    joined = joined[['Id', 'Sales']]
    joined.to_csv('mix.csv')

    return joined
