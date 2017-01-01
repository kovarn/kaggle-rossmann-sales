import logging

import numpy as np
import pandas as pd

from predict_sales.data import linear_features, xgb_features, log_lm_features
from predict_sales.functions import remove_before_changepoint, remove_outliers_lm, allow_modifications, exp_rmspe, predict_elasticnet, xgb_expm1_rmspe, predict_xgboost

# from predict_sales import logger
logger = logging.getLogger(__name__)
pd.set_option('io.hdf.default_format','table')


##
@allow_modifications(True)
def glm_predictions(data: pd.HDFStore, output: pd.HDFStore):
    ##
    logger.info("Dropping store data before changepoint.")
    select_idx = remove_before_changepoint(data, None)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    logger.info("Dropping stores not in test set. Initial shape")
    test_set_stores = data.select_column('test', 'Store').unique()
    idx = data.select_as_coordinates('train', 'Store in test_set_stores')
    select_idx = select_idx.intersection(idx)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    logger.debug("Log transform on sales data")
    idx = data.select_as_coordinates('train', 'Sales > 0')
    select_idx = select_idx.intersection(idx)
    data.put('train_logsales', np.log(data.select('train', 'columns = Sales')), data_columns=True)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    select_idx = remove_outliers_lm(data, select_idx, log_lm_features, test_set_stores)
    logger.info("Removed outliers, reduced shape {0}".format(len(select_idx)))

    ##
    logger.info("Running elasticnet predictions")
    predict_elasticnet(data, output, select_idx, linear_features, test_set_stores,
                                           exp_rmspe, steps=15, step_by=3)

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
def xgb_predictions(data: pd.HDFStore, output: pd.HDFStore):
    ##
    logger.info("Dropping store data before changepoint.")
    select_idx = remove_before_changepoint(data, None)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    logger.info("Dropping stores not in test set. Initial shape")
    test_set_stores = data.select_column('test', 'Store').unique()
    idx = data.select_as_coordinates('train', 'Store in test_set_stores')
    select_idx = select_idx.intersection(idx)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    logger.debug("Log transform on sales data")
    idx = data.select_as_coordinates('train', 'Sales > 0')
    select_idx = select_idx.intersection(idx)
    data.put('train_logsales', np.log(data.select('train', 'columns = Sales')), data_columns=True)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    logger.info("Running xgboost predictions")
    predict_xgboost(data, output, select_idx, xgb_features, xgb_expm1_rmspe,
                                     params=xparams, nrounds=nrounds)


##
def mix_models(output: pd.HDFStore):
    ##
    glm_preds = output.get('glm_predictions')

    xgb_preds = output.get('xgb_predictions')

    assert glm_preds.shape == xgb_preds.shape

    ##
    joined = pd.merge(glm_preds, xgb_preds, how='inner', on='Id')
    joined['Sales'] = 0.985 * (joined['PredictedSales_x'] + joined['PredictedSales_y']) / 2
    assert joined.shape[0] == glm_preds.shape[0]

    ##
    joined = joined[['Id', 'Sales']]
    joined.to_csv('mix.csv', index=False)

    return joined
