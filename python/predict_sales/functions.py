import functools
from collections import namedtuple

import numpy as np
import pandas as pd

from sklearn import linear_model, model_selection

from .data import log_lm_features, check_nulls, make_fold
import logging

import xgboost as xgb

logger = logging.getLogger(__name__)
logger.info("Module loaded")

WRAP = False
logger.info('Wrapping all functions to check df modifications' if WRAP else 'Wrapping disabled')


def allow_modifications(allow):
    def decorate(func):
        if not WRAP:
            return func

        func_name = func.__name__

        @functools.wraps(func)
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


@allow_modifications(True)
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
        train.query('Store != {store} or Date > "{date}"'.format(store=store, date=date), inplace=True)


@allow_modifications(True)
def log_transform_train(train):
    train.query('Sales > 0', inplace=True)
    train['Sales'] = np.log(train['Sales'])


@allow_modifications(False)
def select_features_copy(train, features):
    base_columns = [c for c in ['Sales', 'Id', 'Store', 'Date']
                    if c in train.columns and c not in features]
    columns_to_select = base_columns + list(features)
    return train[columns_to_select]


@allow_modifications(True)
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


@allow_modifications(True)
def remove_outliers_lm(train, features=log_lm_features,
                       z_score=2.5):
    logger.info("{f} train shape {tr}".format(f=remove_outliers_lm.__name__, tr=train.shape))
    # if 'Id' not in train.columns:
    #     logger.info('Adding Id to train data columns.')

    with_id = train
    with_id['Id'] = np.arange(1, train.shape[0] + 1)
    with_fit = predict_lm(select_features(with_id, log_lm_features), None).predicted

    def filter_by_z_score():
        for store_prediction in with_fit:
            errors = abs(store_prediction['PredictedSales'] - store_prediction['Sales'])
            z_scores = errors / errors.median()
            yield z_scores < z_score

    gen = filter_by_z_score()
    return with_id.loc[pd.concat(gen),:].drop('Id', axis=1)


@allow_modifications(False)
def predict_lm(train, test, save_fit=False):
    if test is None:
        logger.info("{f} train shape {tr}, no test".format(f=predict_lm.__name__, tr=train.shape))
    else:
        logger.info(
            "{f} train shape {tr}, test shape {te}".format(f=predict_lm.__name__, tr=train.shape, te=test.shape))
    Predictions = namedtuple('Predictions', 'predicted, fit, store')
    logger.info("Prediction fields {0}".format(Predictions._fields))

    @allow_modifications(True)
    def predict_lm_per_store(store_train, store_test, save_fit=save_fit):
        assert check_nulls(store_train)
        sales = store_train['Sales']
        store_train = store_train.drop(['Date', 'Sales'], axis=1)
        lm = linear_model.LinearRegression()
        fit = lm.fit(store_train, sales)
        store_test_sales = None
        if store_test is None:
            store_test = store_train
            store_test_sales = sales
        pred = fit.predict(store_test[list(store_train.columns)])
        store_test['PredictedSales'] = pred
        if store_test_sales is not None:
            store_test['Sales'] = store_test_sales
        if save_fit:
            return Predictions(predicted=store_test, fit=fit, store=None)
        else:
            return Predictions(predicted=store_test, fit=None, store=None)

    return Predictions(*zip(*predict_per_store(train, test, predict_lm_per_store)))


def glmnet(X, y, family, alpha, nlambda):
    return False


@allow_modifications(False)
def best_glmnet_alpha(store_train, eval_function, steps, predict_interval, step_by, l1_ratio, family, nlambda):
    logger.info("{f} store train shape {tr}".format(f=best_glmnet_alpha.__name__, tr=store_train.shape))
    Predictions = namedtuple('Predictions', 'predictions, actual, scores, fit')
    logger.info("Prediction fields {0}".format(Predictions._fields))

    global_fit = glmnet(store_train.drop(['Sales', 'Date'], axis=1),
                        store_train['Sales'], family=family, alpha=l1_ratio, nlambda=nlambda)
    lambdas = global_fit.glmnet_lambda

    def per_step_map(step):
        fold = make_fold(store_train, step, predict_interval, step_by)
        fold_fit = glmnet(fold.train.drop(['Sales', 'Date'], axis=1), fold.train['Sales'],
                          family=family, alpha=l1_ratio, lambdas=lambdas)
        pred = fold_fit.predict(fold.test.drop(['Id', 'Date'], axis=1), type="response")

        # if (nrow(fold$test) < 2) {
        # predictions < - matrix(0, nrow = nrow(predictions), ncol = ncol(predictions))
        # }

        scores = eval_function(pred, fold.actual['Sales'])

        return Predictions(predictions=pred, actual=fold.actual['Sales'],
                           scores=scores, fit=fold_fit)


def cv_generator(train, steps, predict_interval, step_by):
    for step in range(1, steps + 1):
        yield make_fold(train=train, step=step,
                        predict_interval=predict_interval, step_by=step_by)


@allow_modifications(False)
def predict_elasticnet(train, test=None, with_cv=False, eval_function=None, steps=13,
                       predict_interval=6 * 7, step_by=7,
                       l1_ratio=[.99, 1], family="gaussian",
                       n_alphas=100):
    logger.info("{f} train shape {tr}, test shape {te}"
                .format(f=predict_elasticnet.__name__, tr=train.shape, te=test.shape))
    Predictions = namedtuple('Predictions', 'predicted, fit, errors, store')
    logger.info("Prediction fields {0}".format(Predictions._fields))
    logger.info("Date in train: {0}".format('Date' in train.columns))
    assert family in ("gaussian", "poisson")

    @allow_modifications(True)
    def predict_elasticnet_per_store(store_train, store_test):
        store_train.reset_index(drop=True, inplace=True)
        cv = list(cv_generator(store_train, steps, predict_interval=predict_interval, step_by=step_by))
        en = linear_model.ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas, cv=cv)

        sales = store_train['Sales']
        store = store_train['Store'].iloc[0]
        store_train.drop(['Date', 'Sales'], axis=1, inplace=True)
        store_test_id = store_test['Id']
        store_test.drop(set(store_test.columns).difference(store_train.columns), axis=1, inplace=True)
        fit = en.fit(store_train, sales)
        pred = fit.predict(store_test)
        store_test.drop(set(store_test.columns).difference('Id'), axis=1, inplace=True)
        store_test['PredictedSales'] = pred
        # store_test = pd.concat([store_test_id, pd.Series(pred, name='PredictedSales')], axis=1)
        # # store_test.columns = ['Id', 'PredictedSales']

        cv_median_error = None
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
            logger.info('CV median error {0}'.format(cv_median_error))

        # logger.info('alpha: {alpha}, l1 ratio: {l1_ratio}'
        #             .format(alpha=fit.alpha_, l1_ratio=fit.l1_ratio_))
        # cv = list(cv_generator(store_train, steps, predict_interval=predict_interval, step_by=step_by))
        # scores = model_selection.cross_val_score(fit, X=without_date_sales, y=sales,
        #                                          scoring=None, cv=cv, n_jobs=1, pre_dispatch='n_jobs')
        # logger.info('Median score: {0}'.format(scores.median()))
        return Predictions(predicted=store_test, fit=fit, errors=cv_median_error, store=None)

    return Predictions(*zip(*predict_per_store(train, test, predict_elasticnet_per_store)))


@allow_modifications(False)
def predict_xgboost(train, test, eval_function, params, nrounds):
    logger.info("{f} train shape {tr}, test shape {te}"
                .format(f=predict_xgboost.__name__, tr=train.shape, te=test.shape))
    Predictions = namedtuple('Predictions', 'predicted, fit')
    logger.info("Prediction fields {0}".format(Predictions._fields))

    without_sales = train.drop(['Sales', 'Date'], axis=1)
    dtrain = xgb.DMatrix(without_sales,
                         label=train['Sales'])
    dtest = xgb.DMatrix(test.drop(['Id', 'Date'], axis=1))

    # specify validations set to watch performance
    watchlist = [(dtrain, 'train')]
    fit = xgb.train(params=params, dtrain=dtrain, num_boost_round=nrounds, evals=watchlist,
                    feval=eval_function, early_stopping_rounds=100, maximize=False,
                    verbose_eval=100)

    pred = fit.predict(dtest)
    predicted = test.copy()
    predicted['PredictedSales'] = pred

    return Predictions(predicted=predicted, fit=fit)


def predict_per_store(train, test, predict_fun):
    train_gb = train.groupby('Store')
    test_gb = None
    if test is not None:
        test_gb = test.groupby('Store')

    for s, store_train in train_gb:
        store_test = None
        if test is not None:
            store_test = test_gb.get_group(s)

        yield predict_fun(store_train, store_test)._replace(store=s)


@allow_modifications(True)
def log_revert_predicted(predicted, open):
    predicted['PredictedSales'] = np.exp(predicted['PredictedSales']) * open


def rmspe(predicted, actual):
    assert predicted.shape == actual.shape
    idx = actual > 0
    return np.sqrt(np.square((actual[idx] - predicted[idx]) / actual[idx]).mean())


def exp_rmspe(predicted, actual):
    return rmspe(np.exp(predicted), np.exp(actual))


def getinfo(dtrain, param):
    pass


def xgb_expm1_rmspe(predicted, dtrain):
    return "rmspe", rmspe(np.expm1(predicted), np.expm1(dtrain.get_label()))
