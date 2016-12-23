from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn import linear_model

from .data import log_lm_features, check_nulls
import logging

logger = logging.getLogger(__name__)
logger.info("Module loaded")


def allow_modifications(allow):
    def decorate(func):
        func_name = func.__name__

        def wrapper(*args, **kwargs):
            dfs_with_copy = [(df, df.copy()) for df in args if isinstance(df, pd.DataFrame)]
            return_val = func(*args, **kwargs)
            modified = False
            for df, df_copy in dfs_with_copy:
                if not df.equals(df_copy):
                    modified = True
                    break
            try:
                assert (allow and modified) or (not allow and not modified)
            except AssertionError:
                if allow:
                    logger.warning('{f} does not modify dataframe!'.format(f=func_name))
                else:
                    logger.warning('{f} is modifying dataframe!'.format(f=func_name))
            else:
                if not allow:
                    logger.debug('{f} does not modify dataframe'.format(f=func_name))
                else:
                    logger.debug('{f} is modifying dataframe'.format(f=func_name))
            return return_val

        return wrapper

    return decorate


@allow_modifications(False)
def remove_before_changepoint(train):
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
        train = train.query('Store != {store} or Date > "{date}"'.format(store=store, date=date))
    return train


@allow_modifications(False)
def log_transform_train(train):
    train = train.query('Sales > 0')
    train['Sales'] = np.log(train['Sales'])
    return train


@allow_modifications(False)
def select_features(train, features):
    columns_to_select = ['Sales', 'Id', 'Store', 'Date'] + list(features)
    return train[columns_to_select]


@allow_modifications(False)
def remove_outliers_lm(train, features=log_lm_features,
                       z_score=2.5):
    logger.info("{f} train shape {tr}".format(f=remove_outliers_lm.__name__, tr=train.shape))
    if 'Id' not in train.columns:
        logger.info('Adding Id to train data columns.')

    with_id = train.copy()
    with_id['Id'] = np.arange(1, train.shape[0] + 1)
    with_fit = predict_lm(select_features(with_id, log_lm_features), with_id).predicted

    def filter_by_z_score():
        for store_prediction in with_fit:
            errors = abs(store_prediction['PredictedSales'] - store_prediction['Sales'])
            z_scores = errors / errors.median()
            yield store_prediction[z_scores < z_score].drop('Id', axis=1)

    gen = filter_by_z_score()
    return pd.concat(gen)


Predictions = namedtuple('Predictions', 'predicted, fit, store')


@allow_modifications(False)
def predict_lm(train, test, save_fit=False):
    logger.info("{f} train shape {tr}, test shape {te}".format(f=predict_lm.__name__, tr=train.shape, te=test.shape))

    @allow_modifications(True)
    def predict_lm_per_store(store_train, store_test, save_fit=save_fit):
        assert check_nulls(store_train)
        sales = store_train['Sales']
        store_train = store_train.drop(['Date', 'Sales'], axis=1)
        lm = linear_model.LinearRegression()
        fit = lm.fit(store_train, sales)
        pred = lm.predict(store_test[list(store_train.columns)])
        store_test['PredictedSales'] = pred
        if save_fit:
            return Predictions(predicted=store_test, fit=fit, store=None)
        else:
            return Predictions(predicted=store_test, fit=None, store=None)

    return Predictions(*zip(*predict_per_store(train, test, predict_lm_per_store)))


@allow_modifications(False)
def predict_glmnet(train, test, eval_function, steps=13,
                   predict_interval=6 * 7, step_by=7,
                   alpha=1, family=("gaussian", "poisson"),
                   nlambda=100):
    pass


def predict_per_store(train, test, predict_fun):
    train_gb = train.groupby('Store')
    test_gb = test.groupby('Store')

    for s, store_train in train_gb:
        store_test = test_gb.get_group(s)
        yield predict_fun(store_train, store_test)._replace(store=s)


@allow_modifications(True)
def log_revert_predicted(predicted):
    predicted['PredictedSales'] = np.exp(predicted['PredictedSales']) * predicted['Open']


def rmspe(predicted, actual):
    assert predicted.shape == actual.shape
    idx = actual > 0
    return np.sqrt(np.square((actual[idx] - predicted[idx]) / actual[idx]).mean())


def exp_rmspe(predicted, actual):
    return rmspe(np.exp(predicted), np.exp(actual))
