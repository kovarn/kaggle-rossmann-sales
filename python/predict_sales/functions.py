import functools
import warnings
from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn import linear_model, model_selection
import sklearn

from .data import log_lm_features, check_nulls, make_fold
import logging

import xgboost as xgb
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.info("Module loaded")


def remove_before_changepoint(data: pd.HDFStore, select_idx: pd.Index = None):
    changepoints = {837: '2014-03-16',
                    700: '2014-01-03',
                    681: '2013-06-14',
                    986: '2013-05-22',
                    885: '2014-05-18',
                    589: '2013-05-27',
                    105: '2013-05-20',
                    663: '2013-10-06',
                    764: '2013-04-24',
                    364: '2013-05-31',
                    969: '2013-03-10',
                    803: '2014-01-07',
                    91: '2014-01-14'}
    for store, date in changepoints.items():
        idx = data.select_as_coordinates('train', 'Store != store or Date > pd.Timestamp(date)')
        if select_idx is not None:
            select_idx = select_idx.intersection(idx)
        else:
            select_idx = idx
    return select_idx


def log_transform_train(train):
    train.query('Sales > 0', inplace=True)
    train['Sales'] = np.log(train['Sales'])


def select_features_copy(train, features):
    base_columns = [c for c in ['Sales', 'Id', 'Store', 'Date']
                    if c in train.columns and c not in features]
    columns_to_select = base_columns + list(features)
    return train[columns_to_select]


def select_features_inplace(train, features):
    base_columns = [c for c in ['Sales', 'Id', 'Store', 'Date']
                    if c in train.columns and c not in features]
    columns_to_select = base_columns + list(features)
    columns_to_drop = set(train.columns).difference(columns_to_select)
    train.drop(columns_to_drop, axis=1, inplace=True)


def select_features(train, features, inplace=False):
    if inplace:
        select_features_inplace(train, features)
    else:
        return select_features_copy(train, features)


def remove_outliers_lm(data: pd.HDFStore, select_idx: pd.Index, features, stores,
                       z_score=2.5):
    # logger.info("{f} train shape {tr}".format(f=remove_outliers_lm.__name__, tr=train.shape))
    # if 'Id' not in train.columns:
    #     logger.info('Adding Id to train data columns.')
    def per_store():
        for store in tqdm(stores, desc="Removing outliers"):
            logger.debug("Store {}".format(store))
            store_idx = data.select_as_coordinates('train', 'Store == store')
            store_idx = store_idx.intersection(select_idx)
            sales = data.select('train_logsales', store_idx).set_index(store_idx)['Sales']
            assert sales.shape == (len(store_idx),)

            with_fit = predict_lm_per_store(data, store_idx, log_lm_features, sales)

            errors = abs(with_fit['PredictedSales'] - sales)
            z_scores = errors / errors.median()

            yield store_idx[z_scores < z_score]

    gen = per_store()
    new_select_idx = next(gen)
    for idx in gen:
        new_select_idx = new_select_idx.union(idx)

    return new_select_idx


def predict_lm_per_store(data: pd.HDFStore, select_idx: pd.Index, features, sales, save_fit=False):
    store_train = data.select('train', select_idx, columns=list(features)).set_index(select_idx)
    assert store_train.shape == (len(select_idx), len(features))
    logger.debug('Store train shape {}'.format(store_train.shape))
    logger.debug('Sales shape {}'.format(sales.shape))
    lm = linear_model.LinearRegression()
    fit = lm.fit(store_train, sales)

    pred = fit.predict(store_train)
    store_train['PredictedSales'] = pred
    return store_train


def cv_generator(train, date, steps, predict_interval, step_by):
    for step in range(1, steps + 1):
        yield make_fold(train=train, date=date, step=step,
                        predict_interval=predict_interval, step_by=step_by)


def predict_elasticnet_per_store(data: pd.HDFStore, train_select_idx: pd.Index, test_select_idx: pd.Index,
                                 features, sales, store,
                                 with_cv=False, eval_function=None, steps=13,
                                 predict_interval=6 * 7, step_by=7,
                                 l1_ratio=[.99, 1], family="gaussian", n_alphas=100):
    store_train = data.select('train', train_select_idx, columns=list(features)).set_index(train_select_idx)
    date = data.select_column('train', 'Date')[train_select_idx]

    assert date.shape[0] == store_train.shape[0]
    assert store_train.shape == (len(train_select_idx), len(features))
    logger.debug('Store {0:4d}: train shape {1}, sales shape{2}'
                 .format(int(store), store_train.shape, sales.shape))
    logger.debug(store_train.values.flags)

    cv = list(cv_generator(store_train, date, steps, predict_interval=predict_interval, step_by=step_by))
    en = linear_model.ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas, cv=cv,
                                   n_jobs=-1, selection='random')

    fit = en.fit(store_train, sales)

    logger.debug('Store {0:4d}: alpha {alpha}, l1 ratio {l1_ratio}'
                 .format(int(store), alpha=fit.alpha_, l1_ratio=fit.l1_ratio_))
    # logger.debug('Store {0:4d}: MSE path {1}'.format(int(store), fit.mse_path_))

    if with_cv:
        cv_errors = []
        for fold in cv:
            cv_en = linear_model.ElasticNet(alpha=fit.alpha_, l1_ratio=fit.l1_ratio_)
            cv_train = store_train.iloc[fold.train_idx, :]
            cv_train_sales = sales[fold.train_idx]
            cv_fit = cv_en.fit(cv_train, cv_train_sales)
            cv_test = store_train.iloc[fold.test_idx, :]
            cv_test_sales = sales[fold.test_idx]
            cv_pred = cv_fit.predict(cv_test)
            cv_error = rmspe(np.exp(cv_pred) * cv_test['Open'], np.exp(cv_test_sales))
            cv_errors.append(cv_error)

        cv_median_error = np.median(cv_errors)
        logger.debug('Store {0}. CV errors {1}'.format(store, cv_errors))
        logger.debug('Store {0}. CV median error {1}'.format(store, cv_median_error))

    # cv = list(cv_generator(store_train, steps, predict_interval=predict_interval, step_by=step_by))
    #     scores = model_selection.cross_val_score(fit, X=store_train, y=sales,
    #                                              scoring=None, cv=cv, n_jobs=1, pre_dispatch='n_jobs')
    #     logger.info('Median score: {0}'.format(scores.median()))

    store_test = data.select('test', test_select_idx, columns=list(features))  # type: pd.DataFrame
    store_test_id = store_test.index
    store_test.set_index(test_select_idx, inplace=True)
    assert store_test.shape == (len(test_select_idx), len(features))
    logger.debug('Store test shape {}'.format(store_test.shape))

    pred = fit.predict(store_test)

    result = pd.DataFrame(index=test_select_idx)
    result['Id'] = store_test_id
    result['PredictedSales'] = np.exp(pred) * store_test['Open']
    return result
    # store_test = pd.concat([store_test_id, pd.Series(pred, name='PredictedSales')], axis=1)
    # # store_test.columns = ['Id', 'PredictedSales']


# class GLMPredictions:
#     def __init__(self, data_filename=None):
#         data_filename = data_filename
#         select_idx = None
#         train_key = 'train'
#         label_key = 'train_logsales'
#         stores = None
#         features = None
#         steps = None
#         step_by = None
#         predict_interval = None
#         l1_ratio = None
#         n_alphas = None


def predict_elasticnet(data: pd.HDFStore, output: pd.HDFStore, select_idx: pd.Index, features, stores,
                       with_cv=False, eval_function=None, steps=13,
                       predict_interval=6 * 7, step_by=7,
                       l1_ratio=[.1, .5, .7, .9, .95, .99, 1], family="gaussian",
                       n_alphas=100):
    logger.info("glm predictions. Reading data from {0}".format(data.filename))
    assert family in ("gaussian", "poisson")

    def per_store():
        for store in tqdm(stores, desc="GLM predictions"):
            logger.debug("Store {}".format(store))
            store_train_idx = data.select_as_coordinates('train', 'Store == store')
            store_test_idx = data.select_as_coordinates('test', 'Store == store')
            store_train_idx = store_train_idx.intersection(select_idx)
            sales = data.select('train_logsales', store_train_idx).set_index(store_train_idx)['Sales']
            assert sales.shape == (len(store_train_idx),)

            yield predict_elasticnet_per_store(data, store_train_idx, store_test_idx,
                                               features, sales, store,
                                               with_cv=with_cv, eval_function=eval_function, steps=steps,
                                               predict_interval=predict_interval, step_by=step_by,
                                               l1_ratio=l1_ratio, family=family, n_alphas=n_alphas)

    try:
        output.remove('glm_predictions')
    except KeyError:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=sklearn.exceptions.ConvergenceWarning)
        for preds in per_store():
            output.append('glm_predictions', preds, data_columns=True)
            logger.debug('Wrote predictions, shape {0}'.format(preds.shape))

    result = output.get_storer('glm_predictions')
    logger.info('Wrote predictions, shape{0} to {1}'.format((result.nrows, result.ncols), output.filename))

    # preds = GLMPredictions(data_filename=data.filename)
    # preds.stores = stores
    # preds.select_idx = select_idx
    # preds.output_filename = output.filename
    # preds.features = features
    # preds.steps = steps
    # preds.step_by = step_by
    # preds.predict_interval = predict_interval
    # preds.l1_ratio = l1_ratio
    # preds.n_alphas = n_alphas


def predict_xgboost(data: pd.HDFStore, output: pd.HDFStore, select_idx: pd.Index, features, eval_function, params,
                    nrounds):
    logger.info("xgboost predictions. Reading data from {0}".format(data.filename))

    train = data.select('train', select_idx, columns=list(features))
    logger.info('Train data shape {}'.format(train.shape))
    sales = data.select('train_logsales', select_idx)['Sales']
    logger.info('Sales data shape {}'.format(sales.shape))

    dtrain = xgb.DMatrix(train, label=sales)

    del train, sales

    test = data.select('test', columns=list(features))
    logger.info('Test data shape {}'.format(test.shape))

    dtest = xgb.DMatrix(test)

    del test

    # specify validations set to watch performance
    watchlist = [(dtrain, 'train')]
    fit = xgb.train(params=params, dtrain=dtrain, num_boost_round=nrounds, evals=watchlist,
                    feval=eval_function, early_stopping_rounds=100, maximize=False,
                    verbose_eval=100)

    pred = fit.predict(dtest)

    result = pd.DataFrame(data.select_column('test', 'index'))
    result.columns = ['Id']
    result['PredictedSales'] = np.exp(pred) * data.select_column('test', 'Open')

    output.put('xgb_predictions', result, data_columns=True)
    logger.info('Wrote predictions, shape {0} to {1}'.format(result.shape, output.filename))


def rmspe(predicted, actual):
    assert predicted.shape == actual.shape
    idx = actual > 0
    return np.sqrt(np.square((actual[idx] - predicted[idx]) / actual[idx]).mean())


def exp_rmspe(predicted, actual):
    return rmspe(np.exp(predicted), np.exp(actual))


def xgb_expm1_rmspe(predicted, dtrain):
    return "rmspe", rmspe(np.expm1(predicted), np.expm1(dtrain.get_label()))
