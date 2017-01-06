import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost
from sklearn import linear_model
from sklearn.externals import joblib
from tqdm import tqdm

from .utils.warnings_ import set_warnings_handlers_from, warnings_to_log

logger = logging.getLogger(__name__)
logger.info("Module loaded")

set_warnings_handlers_from(logger)

pd.set_option('io.hdf.default_format', 'table')


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
    # noinspection PyUnusedLocal
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
    def per_store():
        for store in tqdm(stores, desc="Removing outliers"):
            logger.debug("Store {}".format(store))
            store_idx = data.select_as_coordinates('train', 'Store == store')
            store_idx = store_idx.intersection(select_idx)
            sales = data.select('train_logsales', store_idx).set_index(store_idx)['Sales']
            assert sales.shape == (len(store_idx),)

            with_fit = predict_lm_per_store(data, store_idx, features, sales)

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


class DataFromHDF:
    def __init__(self, data_file: str = None, data_store: pd.HDFStore = None,
                 key=None, select_idx: pd.Index = None, columns=None,
                 column=None, data_columns=None):

        if data_store is None:
            data_store = pd.HDFStore(data_file)
        if data_file is None:
            data_file = str(Path(data_store.filename).resolve())

        self.data_file = data_file
        self.data_store = data_store
        self.key = key
        self.select_idx = select_idx
        assert not isinstance(columns, str)
        self.columns = columns
        if columns is not None:
            self.columns = list(columns)
        self.column = column
        self.data_columns = data_columns

        assert isinstance(self.data_store, pd.HDFStore)
        assert isinstance(self.key, str)

    def get(self):
        args = dict(key=self.key, where=self.select_idx, columns=self.columns)
        args = {k: v for k, v in args.items() if v is not None}
        logger.debug("Reading {0!r} data from {1!r}".format(self.key, self.data_store.filename))
        return self.data_store.select(**args)

    def get_index(self):
        return self.get_column('index')

    def get_coordinates(self):
        """
        Useful if select_idx is None
        """
        return self.data_store.select_as_coordinates(self.key)

    def get_column(self, column=None):
        if column is None:
            column = self.column
        try:
            result = self.data_store.select_column(self.key, column)
        except KeyError:
            logger.warning("{0} is probably not a data column. Using select to get data."
                           .format(column))
            if self.select_idx is None:
                return pd.Series(
                    self.data_store.select(self.key, columns=[column]).iloc[:, 0].values,
                    index=self.get_coordinates())
            else:
                return pd.Series(
                    self.data_store.select(self.key, self.select_idx, columns=[column]).iloc[:, 0].values,
                    index=self.select_idx)
        if self.select_idx is None:
            return result
        else:
            return result[self.select_idx]

    def put(self, df: pd.DataFrame):
        args = dict(key=self.key, value=df, data_columns=self.data_columns)
        args = {k: v for k, v in args.items() if v is not None}
        self.data_store.put(**args)
        logger.info('Wrote output, shape {0} to {1!r} in {2!r}'
                    .format(df.shape, self.key, self.data_store.filename))

    def append(self, df: pd.DataFrame):
        args = dict(key=self.key, value=df, data_columns=self.data_columns)
        args = {k: v for k, v in args.items() if v is not None}
        self.data_store.append(**args)
        logger.info('Appended output, shape {0} to {1!r} in {2!r}'
                    .format(df.shape, self.key, self.data_store.filename))

    def subset(self, where=None, copy=True):
        if copy:
            sub = __class__(**self.__dict__)
        else:
            sub = self

        if isinstance(where, __class__):
            sub_select_idx = where.select_idx
        elif isinstance(where, str):
            sub_select_idx = sub.data_store.select_as_coordinates(sub.key, where=where)
        else:
            sub_select_idx = pd.Index(where)

        if sub.select_idx is not None:
            sub.select_idx = sub.select_idx.intersection(sub_select_idx)
        else:
            sub.select_idx = sub_select_idx

        if copy:
            return sub


class XGBPredictions:
    def __init__(self, eval_function=None, params=None, nrounds=100, early_stopping_rounds=100,
                 maximize=False, verbose_eval=100):
        self.verbose_eval = verbose_eval
        self.maximize = maximize
        self.early_stopping_rounds = early_stopping_rounds
        self.nrounds = nrounds
        self.params = params
        self.eval_function = eval_function

        self.model = None  # type: xgboost.Booster

    def fit(self, X: DataFromHDF, y: DataFromHDF):

        logger.info("Fitting model.")

        train = X.get()
        logger.info('Train data shape {}'.format(train.shape))

        sales = y.get_column()
        logger.info('Sales data shape {}'.format(sales.shape))

        dtrain = xgboost.DMatrix(train, label=sales)

        del train, sales

        # specify validations set to watch performance
        watchlist = [(dtrain, X.key)]
        self.model = xgboost.train(params=self.params, dtrain=dtrain, num_boost_round=self.nrounds, evals=watchlist,
                                   feval=self.eval_function, early_stopping_rounds=self.early_stopping_rounds,
                                   maximize=self.maximize, verbose_eval=self.verbose_eval)

        return self

    def predict(self, X: DataFromHDF):

        test = X.get()
        logger.info('Test data shape {}'.format(test.shape))

        dtest = xgboost.DMatrix(test)

        if 'Open' in test.columns:
            open_ = test['Open']
        else:
            open_ = X.get_column('Open')

        del test

        pred = self.model.predict(dtest)

        result = pd.DataFrame(X.get_index())

        result.columns = ['Id']
        result['PredictedSales'] = np.exp(pred) * open_.values

        return result

    def save_model(self, directory_path=None):
        """
        Save the model (the object itself) on disk. Uses joblib for
        serialization.
        :param directory_path: str
        Directory where the file is to be created. Adds a timestamp (in UTC)
        in the file name.
        """

        try:
            Path(directory_path).mkdir(parents=True)
            # Not using exist_ok parameter in order to log directory creation.
        except FileExistsError:
            logger.debug('Saving model to existing directory: {0}.'.format(
                directory_path))
        else:
            logger.info('Created directory {0} for saving model.'.format(
                directory_path))
        from datetime import datetime
        pkl_file_name = datetime.strftime(
            datetime.utcnow(),
            "%Y-%m-%d-%H%M") + '-' + __class__.__name__ + ".pkl"

        pkl_file = str(Path(directory_path, pkl_file_name))
        joblib.dump(self, pkl_file)
        logger.info("Saved model to {0}".format(pkl_file))

    @classmethod
    def load_model(cls, file_name):
        """
        Load the pickled model from disk.
        :param file_name: str
        Name of the file on disk to be read.
        :return: XGBPredictions object
        """
        logger.info("Loading model from {0}".format(file_name))

        loaded_object = joblib.load(file_name)
        if isinstance(loaded_object, cls):
            logger.debug("Loaded model.")
            return loaded_object
        else:
            msg = 'File does not contain a {0} object.'.format(cls)
            logger.error(msg)
            raise TypeError(msg)


class GLMPredictions:
    # noinspection PyDefaultArgument
    def __init__(self, stores=None, steps=15, step_by=3, predict_interval=6 * 7,
                 l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                 n_alphas=100, eval_function=None, n_jobs=-1, selection='random'):
        self.stores = stores
        self.steps = steps
        self.step_by = step_by
        self.predict_interval = predict_interval
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.eval_function = eval_function
        self.n_jobs = n_jobs
        self.selection = selection
        self.models = dict()

    def fit(self, X: DataFromHDF, y: DataFromHDF, stores=None, with_cv=False):

        if stores is None:
            stores = self.stores

        logger.info("Fitting model for {0} stores.".format(len(stores)))

        for store in tqdm(stores, desc="GLM predictions"):
            logger.debug("Store {}".format(store))
            store_X = X.subset('Store == {store}'.format(store=store))

            store_y = y.subset(store_X)

            assert store_X.select_idx.shape == store_y.select_idx.shape

            self.fit_per_store(store_X, store_y, store,
                               with_cv=with_cv)
        return self

    def fit_per_store(self, X: DataFromHDF, y: DataFromHDF, store, with_cv=False):
        store_train = X.get()
        sales = y.get_column()
        date = X.get_column('Date')

        assert date.shape[0] == store_train.shape[0]
        # assert store_train.shape == (len(store_train_idx), len(features))
        logger.debug('Store {0:4d}: train shape {1}, sales shape{2}'
                     .format(int(store), store_train.shape, sales.shape))
        logger.debug(store_train.values.flags)

        cv = list(cv_generator(store_train, date, self.steps, predict_interval=self.predict_interval,
                               step_by=self.step_by))
        en = linear_model.ElasticNetCV(l1_ratio=self.l1_ratio, n_alphas=self.n_alphas, cv=cv,
                                       n_jobs=self.n_jobs, selection=self.selection)

        with warnings_to_log('ConvergenceWarning'):
            fit = en.fit(store_train, sales)

        self.models[store] = fit

        logger.debug('Store {0:4d}: alpha {alpha}, l1 ratio {l1_ratio}'
                     .format(int(store), alpha=fit.alpha_, l1_ratio=fit.l1_ratio_))
        logger.debug('Store {0:4d}: Best MSE {1}'.format(int(store), fit.mse_path_.ravel().min()))

        if with_cv:
            cv_errors = []
            for fold in cv:
                cv_en = linear_model.ElasticNet(alpha=fit.alpha_, l1_ratio=fit.l1_ratio_)
                cv_train = store_train.iloc[fold[0], :]
                cv_train_sales = sales[fold[0]]
                cv_fit = cv_en.fit(cv_train, cv_train_sales)
                cv_test = store_train.iloc[fold[1], :]
                cv_test_sales = sales[fold[1]]
                cv_pred = cv_fit.predict(cv_test)
                cv_error = rmspe(np.exp(cv_pred) * cv_test['Open'], np.exp(cv_test_sales))
                cv_errors.append(cv_error)

            cv_median_error = np.median(cv_errors)
            logger.debug('Store {0}. CV errors {1}'.format(store, cv_errors))
            logger.debug('Store {0}. CV median error {1}'.format(store, cv_median_error))

    def predict(self, X: DataFromHDF, stores=None):

        if stores is None:
            stores = self.stores

        logger.info("Predictions for {0} stores".format(len(stores)))

        def per_store():
            for store in tqdm(stores, desc="GLM predictions"):
                logger.debug("Store {}".format(store))
                store_X = X.subset('Store == {store}'.format(store=store))

                yield self.predict_per_store(store_X, store)

        result = pd.concat(per_store())  # type: pd.DataFrame

        return result

    def predict_per_store(self, X, store):
        store_test = X.get()  # type: pd.DataFrame
        logger.debug('Store test shape {}'.format(store_test.shape))

        pred = self.models[store].predict(store_test)

        result = pd.DataFrame(X.get_index())
        if 'Open' in store_test.columns:
            open_ = store_test['Open']
        else:
            open_ = X.get_column('Open')

        result.columns = ['Id']
        result['PredictedSales'] = np.exp(pred) * open_.values
        return result

    def save_model(self, directory_path=None):
        """
        Save the model (the object itself) on disk. Uses joblib for
        serialization.
        :param directory_path: str
        Directory where the file is to be created. Adds a timestamp (in UTC)
        in the file name.
        """

        try:
            Path(directory_path).mkdir(parents=True)
            # Not using exist_ok parameter in order to log directory creation.
        except FileExistsError:
            logger.debug('Saving model to existing directory: {0}.'.format(
                directory_path))
        else:
            logger.info('Created directory {0} for saving model.'.format(
                directory_path))
        from datetime import datetime
        pkl_file_name = datetime.strftime(
            datetime.utcnow(),
            "%Y-%m-%d-%H%M") + '-' + __class__.__name__ + ".pkl"

        pkl_file = str(Path(directory_path, pkl_file_name))
        joblib.dump(self, pkl_file)
        logger.info("Saved model to {0}".format(pkl_file))

    @classmethod
    def load_model(cls, file_name):
        """
        Load the pickled model from disk.
        :param file_name: str
        Name of the file on disk to be read.
        :return: GLMPredictions object
        """
        logger.info("Loading model from {0}".format(file_name))

        loaded_object = joblib.load(file_name)
        if isinstance(loaded_object, cls):
            logger.debug("Loaded model.")
            return loaded_object
        else:
            msg = 'File does not contain a {0} object.'.format(cls)
            logger.error(msg)
            raise TypeError(msg)


def rmspe(predicted, actual):
    assert predicted.shape == actual.shape
    idx = actual > 0
    return np.sqrt(np.square((actual[idx] - predicted[idx]) / actual[idx]).mean())


def exp_rmspe(predicted, actual):
    return rmspe(np.exp(predicted), np.exp(actual))


def xgb_expm1_rmspe(predicted, dtrain):
    return "rmspe", rmspe(np.expm1(predicted), np.expm1(dtrain.get_label()))


def make_fold(train, date, step, predict_interval, step_by):
    train = train.reset_index(drop=True)
    dates = pd.date_range(date.min(), date.max())
    total = dates.shape[0]
    last_train = total - predict_interval - (step - 1) * step_by
    last_train_date = dates[last_train - 1]
    last_predict = last_train + predict_interval
    last_predict_date = dates[last_predict - 1]
    train_idx = train.index[date <= last_train_date]
    test_idx = train.index[(date > last_train_date) & (date <= last_predict_date)]
    return train_idx, test_idx
