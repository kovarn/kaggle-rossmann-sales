import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from predict_sales.data import linear_features, xgb_features, log_lm_features
from predict_sales.functions import remove_before_changepoint, remove_outliers_lm, xgb_expm1_rmspe, GLMPredictions, \
    DataFromHDF, XGBPredictions
from predict_sales.utils.warnings_ import warnings_to_log

logger = logging.getLogger(__name__)
# +-from predict_sales import logger

pd.set_option('io.hdf.default_format', 'table')


##
# ++output_dir = Path('..','output').resolve()
# ++data = pd.HDFStore(str(output_dir / 'data.h5'))
# ++output = pd.HDFStore(str(output_dir / 'output.h5'))
# ++model_save_dir = str(output_dir)


##
def chunks(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


##
def get_saved_glm_model(model_dir):
    if Path(model_dir).is_dir():
        model_file = sorted(Path(model_dir).glob('*GLMPredictions.pkl'), reverse=True)[0]
    else:
        model_file = model_dir
    return GLMPredictions.load_model(str(model_file.resolve()))


##
def get_saved_xgb_model(model_dir):
    if Path(model_dir).is_dir():
        model_file = sorted(Path(model_dir).glob('*XGBPredictions.pkl'), reverse=True)[0]
    else:
        model_file = model_dir
    return XGBPredictions.load_model(str(model_file.resolve()))


##
def glm_predictions(data: pd.HDFStore, output: pd.HDFStore,
                    model_save_dir=None, predict_train=True, from_saved_model=False):
    # +-
    test_set_stores = data.select_column('test', 'Store').unique()
    ##
    if from_saved_model:
        if from_saved_model is True:
            glm = get_saved_glm_model(model_save_dir)
        else:
            glm = get_saved_glm_model(from_saved_model)

    else:

        ##
        logger.info("Dropping store data before changepoint.")
        select_idx = remove_before_changepoint(data, None)
        logger.info("Reduced to {0}".format(len(select_idx)))

        ##
        logger.info("Dropping stores not in test set. Initial shape")
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
        logger.info("Running glm training")
        X = DataFromHDF(data_store=data, key='train', select_idx=select_idx, columns=linear_features)
        y = DataFromHDF(data_store=data, key='train_logsales', select_idx=select_idx, column='Sales')
        glm = GLMPredictions(stores=test_set_stores, steps=15, step_by=3)
        glm.fit(X, y)

        ##
        if model_save_dir:
            glm.save_model(model_save_dir)

    ##
    logger.info("glm predictions on test set")
    X = DataFromHDF(data_store=data, key='test', columns=linear_features)
    glm_output = DataFromHDF(data_store=output, key='test/glm', data_columns=True)
    preds = glm.predict(X)
    glm_output.put(preds)

    ##
    if predict_train:
        logger.info("glm predictions on training set")
        X = DataFromHDF(data_store=data, key='train', columns=linear_features)
        glm_output = DataFromHDF(data_store=output, key='train/glm', data_columns=True)
        preds = glm.predict(X)
        glm_output.put(preds)


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
def xgb_predictions(data: pd.HDFStore, output: pd.HDFStore,
                    model_save_dir=None, predict_train=True, from_saved_model=False):
    # +-
    ##
    # noinspection PyUnusedLocal
    test_set_stores = data.select_column('test', 'Store').unique()

    if from_saved_model:
        if from_saved_model is True:
            xgb = get_saved_xgb_model(model_save_dir)
        else:
            xgb = get_saved_xgb_model(from_saved_model)

    else:

        logger.info("Dropping store data before changepoint.")
        select_idx = remove_before_changepoint(data, None)
        logger.info("Reduced to {0}".format(len(select_idx)))

        ##
        logger.info("Dropping stores not in test set. Initial shape")
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
        logger.info("Running xgboost training")
        X = DataFromHDF(data_store=data, key='train', select_idx=select_idx, columns=xgb_features)
        y = DataFromHDF(data_store=data, key='train_logsales', select_idx=select_idx, column='Sales')
        xgb = XGBPredictions(eval_function=xgb_expm1_rmspe, params=xparams, nrounds=3000)
        xgb.fit(X, y)

        ##
        if model_save_dir:
            xgb.save_model(model_save_dir)

    ##
    logger.info("xgboost predictions on test set")
    X = DataFromHDF(data_store=data, key='test', columns=xgb_features)
    xgb_output = DataFromHDF(data_store=output, key='test/xgb', data_columns=True)
    preds = xgb.predict(X)
    xgb_output.put(preds)

    ##
    if predict_train:
        logger.info("xgboost predictions on training set")
        xgb_output = DataFromHDF(data_store=output, key='train/xgb', data_columns=True)
        select_idx = data.select_as_coordinates('train', 'Store in test_set_stores')
        X = DataFromHDF(data_store=data, key='train', select_idx=select_idx, columns=xgb_features)
        predict_in_chunks(xgb, X, xgb_output)


def predict_in_chunks(model, X, preds_output):
    first = True
    for chunk_idx in tqdm(chunks(X.select_idx, 100000), desc='Chunksize: 100,000 rows'):
        preds = model.predict(X.subset(chunk_idx))
        if first:
            preds_output.put(preds)
            first = False
        else:
            preds_output.append(preds)


##
def mix_models(output: pd.HDFStore, result_file):
    # +-
    ##
    glm_preds = output.get('test/glm')

    xgb_preds = output.get('test/xgb')

    assert glm_preds.shape[1] == xgb_preds.shape[1]

    if glm_preds.shape[0] != xgb_preds.shape[0]:
        logger.warning(
            'glm and xgb predictions in {0!r} have different lengths: {1}, {2}'
                .format(result_file, glm_preds.shape[0], xgb_preds.shape[0]))

    ##
    joined = pd.merge(glm_preds, xgb_preds, how='inner', on='Id')
    joined['Sales'] = 0.985 * (joined['PredictedSales_x'] + joined['PredictedSales_y']) / 2
    assert joined.shape[0] == glm_preds.shape[0]

    joined = joined[['Id', 'Sales']]

    ##
    joined.to_csv(result_file, index=False)

    return joined


# +-
##
if __name__ == '__main__':
    from predict_sales import logger

    output_dir = Path('..', '..', 'output').resolve()
    result_file = str(output_dir / 'mix.csv')
    data = pd.HDFStore(str(output_dir / 'data.h5'))
    output = pd.HDFStore(str(output_dir / 'output.h5'))
    model_save_dir = str(output_dir)

    glm_predictions(data, output, model_save_dir)

    xgb_predictions(data, output, model_save_dir)

    mix_models(output, result_file)

    data.close()
    output.close()
