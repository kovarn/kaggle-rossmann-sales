import logging
from pathlib import Path

import numpy as np
import pandas as pd

from predict_sales.data import linear_features, xgb_features, log_lm_features
from predict_sales.functions import remove_before_changepoint, remove_outliers_lm, xgb_expm1_rmspe, predict_xgboost, \
    GLMPredictions
from predict_sales.utils.warnings_ import set_warnings_handlers_from, warnings_to_log

logger = logging.getLogger(__name__)
# +-from predict_sales import logger

set_warnings_handlers_from(logger)

pd.set_option('io.hdf.default_format', 'table')


##
# ++output_dir = Path('..','output').resolve()
# ++data = pd.HDFStore(str(output_dir / 'data.h5'))
# ++output = pd.HDFStore(str(output_dir / 'output.h5'))
# ++model_save_dir = str(output_dir)


##
def glm_predictions(data: pd.HDFStore, output: pd.HDFStore, model_save_dir=None):
    # +-
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
    with warnings_to_log('divide by zero'):
        data.put('train_logsales', np.log(data.select('train', 'columns = Sales')), data_columns=True)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    select_idx = remove_outliers_lm(data, select_idx, log_lm_features, test_set_stores)
    logger.info("Removed outliers, reduced shape {0}".format(len(select_idx)))

    ##
    logger.info("Running elasticnet predictions")
    glm = GLMPredictions(stores=test_set_stores, steps=15, step_by=3)
    glm.fit(features=linear_features, data_store=data, train_key='train',
            train_idx=select_idx, label_key='train_logsales')

    ##
    glm.predict(features=linear_features, data_store=data, test_key='test',
                output_store=output)

    ##
    if model_save_dir:
        glm.save_model(model_save_dir)


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
def xgb_predictions(data: pd.HDFStore, output: pd.HDFStore):
    # +-
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
    with warnings_to_log('divide by zero'):
        data.put('train_logsales', np.log(data.select('train', 'columns = Sales')), data_columns=True)
    logger.info("Reduced to {0}".format(len(select_idx)))

    ##
    logger.info("Running xgboost predictions")
    predict_xgboost(data, output, select_idx, xgb_features, xgb_expm1_rmspe,
                    params=xparams, nrounds=nrounds)


##
def mix_models(output: pd.HDFStore):
    # +-
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
