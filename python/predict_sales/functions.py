from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn import linear_model

from .data import log_lm_features
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
    if 'Id' not in train.columns:
        logger.info('Adding Id to train data columns.')

    train['Id'] = np.arange(1, train.shape[0] + 1)
    train = select_features(train, features)
    # with_fit < - predict_lm(select_features(with_id, log_lm_features),
    #                         with_id)$predicted
    # with_fit % > %
    # group_by(Store) % > %
    # mutate(Error=abs(PredictedSales - Sales),
    #        ZScore=Error / median(Error)) % > %
    # filter(ZScore < 2.5) % > %
    # select(-Id, -Error, -ZScore) % > %
    # ungroup


predict_fit = namedtuple('predict_fit', 'predict, fit')


@allow_modifications(False)
def predict_lm_per_store(store_train, store_test, save_fit):
    lm = linear_model.LinearRegression()
    fit = lm.fit(store_train.drop(['Date', 'Sales'], axis=1), store_train['Sales'])
    pred = lm.predict(store_test)
    store_test['PredictedSales'] = pred
    if save_fit:
        return predict_fit(predicted=store_test, fit=fit)
    else:
        return predict_fit(predicted=store_test, fit=None)


@allow_modifications(False)
def predict_lm(train, test, save_fit=False):
    predict_lm_per_store(train, test, save_fit)
