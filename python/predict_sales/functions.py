import logging
import warnings
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn import linear_model
from tqdm import tqdm

from .data import log_lm_features
from .utils.warnings_ import set_warnings_handlers_from, warnings_to_log

logger = logging.getLogger(__name__)
logger.info("Module loaded")

set_warnings_handlers_from(logger)


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


class DataFromHDF:
    def __init__(self, data_file: str = None, data_store: pd.HDFStore = None,
                 key=None, select_idx: pd.Index = None, columns=None):
        if data_store is None:
            data_store = pd.HDFStore(data_file)
        if data_file is None:
            data_file = str(Path(data_store.filename).resolve())

        self.data_file = data_file
        self.data_store = data_store
        self.key = key
        self.select_idx = select_idx
        self.columns = list(columns)

        assert isinstance(self.data_store, pd.HDFStore)
        assert isinstance(self.key, str)

    def get(self, sub_idx=None):
        args = dict(key=self.key, where=self.select_idx, columns=self.columns)
        args = {k: v for k, v in args.items() if v is not None}

        return self.data_store.select(**args)


class XGBPredictions:
    def __init__(self, eval_function=None, params=None, nrounds=100, early_stopping_rounds=100,
                 maximize=False, verbose_eval=100):
        self.verbose_eval = verbose_eval
        self.maximize = maximize
        self.early_stopping_rounds = early_stopping_rounds
        self.nrounds = nrounds
        self.params = params
        self.eval_function = eval_function

        self.model = None  # type: xgb.Booster

        self._fit_params = None

    def fit(self, features, data_file: str = None, data_store: pd.HDFStore = None, train_key='train',
            train_idx: pd.Index = None,
            label_key='train_logsales'):
        assert data_store is None or data_file is None
        if data_file is not None:
            data_store = pd.HDFStore(data_file)
        else:
            data_file = str(Path(data_store.filename).resolve())

        self._fit_params = locals().copy()
        # Remove those items that can cause problem while pickling
        self._fit_params.pop('self')
        self._fit_params.pop('data_store')

        logger.debug("Fit params \n {0!r}".format(self._fit_params))
        logger.info("Fitting model.")
        logger.info("Reading {0!r} data from {1!r}".format(train_key, data_store.filename))

        train = data_store.select(train_key, train_idx, columns=list(features))
        logger.info('Train data shape {}'.format(train.shape))
        sales = data_store.select(label_key, train_idx)['Sales']
        logger.info('Sales data shape {}'.format(sales.shape))

        dtrain = xgb.DMatrix(train, label=sales)

        del train, sales

        # specify validations set to watch performance
        watchlist = [(dtrain, train_key)]
        self.model = xgb.train(params=self.params, dtrain=dtrain, num_boost_round=self.nrounds, evals=watchlist,
                               feval=self.eval_function, early_stopping_rounds=self.early_stopping_rounds,
                               maximize=self.maximize, verbose_eval=self.verbose_eval)

        return self

    def predict(self, features, data_file: str = None, data_store: pd.HDFStore = None,
                test_key='test', test_idx: pd.Index = None,
                output_file: str = None, output_store: pd.HDFStore = None,
                output_key='test/xgb'):

        assert data_store is None or data_file is None
        if data_store is None:
            data_store = pd.HDFStore(data_file)

        logger.info("Predictions on {0!r} from {1!r}"
                    .format(test_key, data_store.filename))

        assert output_store is None or output_file is None
        if output_file is not None:
            output_store = pd.HDFStore(output_file)

            logger.info("Will save predictions to {0!r} in {1!r}"
                        .format(output_store.filename, output_key))

        if test_idx is None:
            test = data_store.select(test_key, columns=list(features))
        else:
            test = data_store.select(test_key, test_idx, columns=list(features))
        logger.info('Test data shape {}'.format(test.shape))

        dtest = xgb.DMatrix(test)

        del test

        pred = self.model.predict(dtest)

        if test_idx is None:
            result = pd.DataFrame(data_store.select_column(test_key, 'index'))
            open_ = data_store.select_column(test_key, 'Open')
        else:
            result = pd.DataFrame(data_store.select_column(test_key, 'index')[test_idx])
            open_ = data_store.select_column(test_key, 'Open')[test_idx]

        result.columns = ['Id']
        result['PredictedSales'] = np.exp(pred) * open_

        if output_store:
            try:
                output_store.remove(output_key)
            except KeyError:
                pass

            output_store.put(output_key, result, data_columns=True)
            logger.info('Wrote predictions, shape {0} to {1!r} in {2!r}'
                        .format(result.shape, output_key, output_store.filename))

        else:
            return result


class GLMPredictions:
    def __init__(self, stores=None, steps=15, step_by=3, predict_interval=6 * 7,
                 l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                 n_alphas=100, eval_function=None, njobs=-1, selection='random'):
        self.stores = stores
        self.steps = steps
        self.step_by = step_by
        self.predict_interval = predict_interval
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.eval_function = eval_function
        self.njobs = njobs
        self.selection = selection
        self.models = dict()

        self._fit_params = None

    def fit(self, features, data_file: str = None, data_store: pd.HDFStore = None, train_key='train',
            train_idx: pd.Index = None,
            label_key='train_logsales', stores=None, with_cv=False):
        assert data_store is None or data_file is None
        if data_file is not None:
            data_store = pd.HDFStore(data_file)
        else:
            data_file = str(Path(data_store.filename).resolve())

        if stores is None:
            stores = self.stores

        self._fit_params = locals().copy()
        # Remove those items that can cause problem while pickling
        self._fit_params.pop('self')
        self._fit_params.pop('data_store')

        logger.debug("Fit params \n {0!r}".format(self._fit_params))
        logger.info("Fitting model for {0} stores.".format(len(stores)))
        logger.info("Reading {0!r} data from {1!r}".format(train_key, data_store.filename))

        for store in tqdm(stores, desc="GLM predictions"):
            logger.debug("Store {}".format(store))
            store_train_idx = data_store.select_as_coordinates(train_key, 'Store == store')
            # store_test_idx = data_store.select_as_coordinates('test', 'Store == store')
            if train_idx is not None:
                store_train_idx = store_train_idx.intersection(train_idx)
            sales = data_store.select(label_key, store_train_idx).set_index(store_train_idx)['Sales']
            assert sales.shape == (len(store_train_idx),)

            self.fit_per_store(features, data_store, train_key, store_train_idx,
                               sales, store,
                               with_cv=with_cv, eval_function=self.eval_function, steps=self.steps,
                               predict_interval=self.predict_interval, step_by=self.step_by,
                               l1_ratio=self.l1_ratio, n_alphas=self.n_alphas, n_jobs=self.njobs,
                               selection=self.selection)
        return self

    def fit_per_store(self, features, data_store, train_key, store_train_idx, sales, store, with_cv, eval_function,
                      steps,
                      predict_interval, step_by, l1_ratio, n_alphas, n_jobs, selection):
        store_train = (data_store.select(train_key, store_train_idx, columns=list(features))
                       .set_index(store_train_idx))
        date = data_store.select_column(train_key, 'Date')[store_train_idx]

        assert date.shape[0] == store_train.shape[0]
        assert store_train.shape == (len(store_train_idx), len(features))
        logger.debug('Store {0:4d}: train shape {1}, sales shape{2}'
                     .format(int(store), store_train.shape, sales.shape))
        logger.debug(store_train.values.flags)

        cv = list(cv_generator(store_train, date, steps, predict_interval=predict_interval, step_by=step_by))
        en = linear_model.ElasticNetCV(l1_ratio=l1_ratio, n_alphas=n_alphas, cv=cv,
                                       n_jobs=n_jobs, selection=selection)

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

    def predict(self, features, data_file: str = None, data_store: pd.HDFStore = None,
                test_key='test', test_idx: pd.Index = None,
                output_file: str = None, output_store: pd.HDFStore = None,
                output_key='test/glm', stores=None):

        assert data_store is None or data_file is None
        if data_store is None:
            data_store = pd.HDFStore(data_file)

        if stores is None:
            stores = self.stores

        logger.info("Predictions for {0} stores on {1!r} from {2!r}"
                    .format(len(stores), test_key, data_store.filename))

        assert output_store is None or output_file is None
        if output_file is not None:
            output_store = pd.HDFStore(output_file)

            logger.info("Will save predictions to {0!r} in {1!r}"
                        .format(output_store.filename, output_key))

        def per_store():
            for store in tqdm(stores, desc="GLM predictions"):
                logger.debug("Store {}".format(store))
                store_test_idx = data_store.select_as_coordinates(test_key, 'Store == store')
                if test_idx is not None:
                    store_test_idx = store_test_idx.intersection(test_idx)

                yield self.predict_per_store(features, data_store, test_key, store_test_idx, store)

        if output_store:
            try:
                output_store.remove(output_key)
            except KeyError:
                pass

            for preds in per_store():
                output_store.append(output_key, preds, data_columns=True)
                logger.debug('Wrote predictions, shape {0}'.format(preds.shape))

            result = output_store.get_storer(output_key)
            logger.info('Wrote predictions, shape{0} to {1!r} in {2!r}'
                        .format((result.nrows, result.ncols),
                                output_key, output_store.filename))

        else:
            return pd.concat(per_store())

    def predict_per_store(self, features, data_store, test_key, store_test_idx, store):
        store_test = data_store.select(test_key, store_test_idx, columns=list(features))  # type: pd.DataFrame
        store_test_id = store_test.index
        store_test.set_index(store_test_idx, inplace=True)
        assert store_test.shape == (len(store_test_idx), len(features))
        logger.debug('Store test shape {}'.format(store_test.shape))

        pred = self.models[store].predict(store_test)

        result = pd.DataFrame(index=store_test_idx)
        result['Id'] = store_test_id
        result['PredictedSales'] = np.exp(pred) * store_test['Open']
        return result

    def save_model(self, directory_path=None):
        """
        Save the model (the object itself) on disk. Uses pickle for
        serialization.
        :param directory_path: str
        Directory where the file is to be created. Adds a timestamp (in UTC)
        in the file name.
        """
        import pickle
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

        with Path(directory_path, pkl_file_name).open(mode='wb') as pkl_file:
            pickle.dump(self, pkl_file)
        logger.info(
            "Saved model to {0}".format(Path(directory_path, pkl_file_name)))

    @classmethod
    def load_model(cls, file_name):
        """
        Load the pickled model from disk.
        :param file_name: str
        Name of the file on disk to be read.
        :return: FixPointClassifier object
        """
        logger.info("Loading model from {0}".format(file_name))
        import pickle
        with open(file_name, mode='rb') as pkl_file:
            loaded_object = pickle.load(pkl_file)
        if isinstance(loaded_object, cls):
            logger.debug("Loaded model.")
            return loaded_object
        else:
            msg = 'File does not contain a {0} object.'.format(cls)
            logger.error(msg)
            raise TypeError(msg)


@warnings_to_log('ConvergenceWarning')
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


def predict_xgboost(data: pd.HDFStore, output: pd.HDFStore, select_idx: pd.Index,
                    features, eval_function, params,
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


# cross_val_fold = namedtuple('cross_val_fold', 'train_idx test_idx')

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
