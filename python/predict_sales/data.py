##
import logging
from collections import OrderedDict, namedtuple
from functools import partial
from itertools import product, chain, starmap

import numpy as np
import pandas as pd

# ToDo: Remove this
REDUCE_DATA = 20
SMALL_TRAIN = False

logger = logging.getLogger(__name__)
logger.info("Module loaded")

pd.set_option('float_format', "{0:.2f}".format)

##
# md
'''
Feature names
'''
##
fourier_terms = 5
fourier_names = ['Fourier' + str(x) for x in range(1, 2 * fourier_terms + 1)]

##
base_linear_features = ["Promo", "Promo2Active", "SchoolHoliday",
                        "DayOfWeek1", "DayOfWeek2", "DayOfWeek3",
                        "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
                        "StateHolidayA", "StateHolidayB", "StateHolidayC",
                        "CompetitionOpen", "Open"]

trend_features = ["DateTrend", "DateTrendLog"]

decay_features = ["PromoDecay", "Promo2Decay",
                  "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                  "StateHolidayBNextDecay"]

log_decay_features = list(map("{0}Log".format, decay_features))

##
stairs_steps = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 21, 28)

stairs_features = list(chain(
    ["Opened", "PromoStarted", "Promo2Started", "TomorrowClosed",
     "WasClosedOnSunday"],
    map("PromoStartedLastDate{0}after".format, (2, 3, 4)),
    starmap("{1}{0}before".format,
            product(stairs_steps, ("StateHolidayCNextDate", "StateHolidayBNextDate",
                                   "StateHolidayANextDate", "LongClosedNextDate"))),
    starmap("{1}{0}after".format,
            product(stairs_steps, ("StateHolidayCLastDate", "StateHolidayBLastDate",
                                   "Promo2StartedDate", "LongOpenLastDate")))
))

##
month_day_features = map("MDay{0}".format, range(1, 32))

month_features = map("Month{0}".format, range(1, 13))

##
linear_features = list(chain(base_linear_features, trend_features, decay_features,
                             log_decay_features, stairs_features, fourier_names,
                             month_day_features, month_features))

glm_features = ("Promo", "SchoolHoliday",
                "DayOfWeek1", "DayOfWeek2", "DayOfWeek3",
                "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
                "StateHolidayA", "StateHolidayB", "StateHolidayC",
                "CompetitionOpen", "PromoDecay", "Promo2Decay",
                "DateTrend", "DateTrendLog",
                "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                "Fourier1", "Fourier2", "Fourier3", "Fourier4")

log_lm_features = ("Promo", "Promo2Active", "SchoolHoliday",
                   "DayOfWeek1", "DayOfWeek2", "DayOfWeek3",
                   "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
                   "StateHolidayA", "StateHolidayB", "StateHolidayC",
                   "CompetitionOpen", "PromoDecay", "Promo2Decay",
                   "DateTrend", "DateTrendLog",
                   "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                   "Fourier1", "Fourier2", "Fourier3", "Fourier4")

categorical_numeric_features = list(chain(
    ("Store", "Promo", "Promo2Active", "SchoolHoliday", "DayOfWeek",
     "StateHolidayN", "CompetitionOpen", "StoreTypeN", "AssortmentN",
     "DateTrend", "MDay", "Month", "Year"), decay_features, stairs_features,
    fourier_names
))

xgb_features = list(chain(("Store", "DayOfWeek", "Open", "Promo", "SchoolHoliday",
                           "StoreTypeN", "AssortmentN", "CompetitionDistance",
                           "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
                           "Promo2", "Promo2SinceWeek", "Promo2SinceYear",
                           "PromoIntervalN", "Month", "Year", "MDay"),
                          decay_features, stairs_features, fourier_names))


##
def check_nulls(df, column=None):
    if column:
        df = df[column]
    anynulls = pd.isnull(df).any()
    if isinstance(anynulls, bool):
        return not anynulls
    return not anynulls.any()


##
def get_date(row):
    """
    Combines the year and month values to return a date
    :param row: DataFrame row
    :return: Timestamp for the date
    """
    year, month = row[['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']]
    if not pd.isnull(year):
        return pd.Timestamp(int(year), int(month), 1)


##
# Generate Fourier terms

def fourier(ts_length, period, terms):
    """
    Generate periodic terms with the given period
    :param ts_length: The length of the generated terms (corresponding to the
    time series length)
    :param period: The period
    :param terms: The number of series to generate.
    :return: DataFrame of shape ts_length, 2*terms. Columns are named
    s1, c1, s2, c2, ..., s<terms>, c<terms>
    """
    A = pd.DataFrame()
    fns = OrderedDict([('s', np.sin), ('c', np.cos)])
    terms_range = range(1, terms + 1)
    for term, (fn_name, fn) in product(terms_range, fns.items()):
        A[fn_name + str(term)] = fn(2 * np.pi * term * np.arange(1, ts_length + 1) / period)
    logger.info("Created fourier terms of shape {0}".format(A.shape))
    return A


##
# ToDo: Check interpretation
month_to_nums_dict = {'Jan,Apr,Jul,Oct': frozenset((1, 4, 7, 10)),
                      'Feb,May,Aug,Nov': frozenset((2, 5, 8, 11)),
                      'Mar,Jun,Sept,Dec': frozenset((3, 6, 9, 12))}


def month_to_nums(months):
    if months:
        return month_to_nums_dict[months]


# 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval', 'Date'
def is_promo2_active(row):
    if row[0]:
        start_year = row[1]
        start_week = row[2]
        interval = row[3]
        date = row[4]
        if ((date.year > start_year)
            or ((date.year == start_year) and (date.week >= start_week))):
            if date.month in interval:
                return 1
    return 0

def is_promo2_active2(row):
    start_year = row[1]
    start_week = row[2]
    interval = row[3]
    date = row[4]
    if ((date.year > start_year)
        or ((date.year == start_year) and (date.week >= start_week))):
        if date.month in interval:
            return 1
    return 0

##
def make_decay_features(g, promo_after, promo2_after, holiday_b_before,
                        holiday_c_before, holiday_c_after):
    g['PromoDecay'] = g['Promo'] * np.maximum(
        promo_after - (g['Date'] - g['PromoStartedLastDate']).dt.days
        , 0)
    g['StateHolidayCLastDecay'] = (1 - g['StateHolidayC']) * np.maximum(
        holiday_c_after - (g['Date'] - g['StateHolidayCLastDate']).dt.days
        , 0)
    g['Promo2Decay'] = g['Promo2Active'] * np.maximum(
        promo2_after - (g['Date'] - g['Promo2StartedDate']).dt.days
        , 0)
    g['StateHolidayBNextDecay'] = (1 - g['StateHolidayB']) * np.maximum(
        holiday_b_before - (g['StateHolidayBNextDate'] - g['Date']).dt.days
        , 0)
    g['StateHolidayCNextDecay'] = (1 - g['StateHolidayC']) * np.maximum(
        holiday_c_before - (g['StateHolidayCNextDate'] - g['Date']).dt.days
        , 0)


##
def scale_log_features(g, *features):
    for f in features:
        g[f] /= g[f].max()
        g["{0}Log".format(f)] = np.log1p(g[f])
        g["{0}Log".format(f)] /= g["{0}Log".format(f)].max()


##
def make_before_stairs(g, *features, days=(2, 3, 4, 5, 7, 14, 28)):
    for f in features:
        before = (g[f] - g['Date']).dt.days
        for d in days:
            g["{0}{1}before".format(f, d)] = before.apply(
                lambda s: 1 if 0 <= s < d else 0)


##
def make_after_stairs(g, *features, days):
    for f in features:
        since = (g['Date'] - g[f]).dt.days
        for d in days:
            g["{0}{1}after".format(f, d)] = since.apply(
                lambda s: 1 if 0 <= s < d else 0)


##
class Data:
    ##
    train = None
    test = None
    small_train = None
    small_fold = None
    one_train = None
    one_test = None

    @classmethod
    def save(cls, train_file, test_file):
        import pickle
        with open(train_file, mode='wb') as pkl_file:
            pickle.dump(cls.train, pkl_file)
            logger.info('Saved train data to {0}'.format(train_file))
        with open(test_file, mode='wb') as pkl_file:
            pickle.dump(cls.test, pkl_file)
            logger.info('Saved test data to {0}'.format(test_file))

    ##
    @classmethod
    def process_input(cls, store_file, train_file, test_file):
        logger.info("Start processing input **************************")
        ##
        store = pd.read_csv(store_file,
                            parse_dates={
                                'CompetitionOpenDate': ['CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth']},
                            date_parser=partial(pd.to_datetime, format="%Y %m", errors='coerce'),
                            converters={'PromoInterval': month_to_nums}
                            )
        logger.info("Read store data: {0} rows, {1} columns".format(*store.shape))

        if REDUCE_DATA:
            logger.info("DROPPING DATA FOR DEBUGGING")
            store = store.iloc[:REDUCE_DATA, :]
            logger.info("Store data reduced to {0} rows".format(store.shape[0]))

        ##
        train_csv = pd.read_csv(train_file, dtype={'StateHoliday': 'str'}, parse_dates=['Date'])
        logger.info("Read train data: {0} rows, {1} columns".format(*train_csv.shape))

        ##
        train_csv['Open'] = train_csv['Sales'].astype(bool).astype(int)
        assert check_nulls(train_csv, 'Open')
        logger.debug("Patch train_csv Open")

        ##
        # Generate DatetimeIndex for the range of dates in the training set
        train_range_lims = (train_csv['Date'].min(), train_csv['Date'].max())
        train_range = pd.date_range(train_range_lims[0], train_range_lims[1], name='Date')
        logger.info("Training range is {0} - {1}".format(train_range.min(), train_range.max()))

        ##
        # Fill in gaps in dates for each store

        def fill_gaps_by_store(g):
            s = g['Store'].iloc[0]
            logger.debug("Store {s} initial shape {shape}"
                         .format(s=s, shape=g.shape))
            filled = (g.set_index('Date')
                      .reindex(train_range)
                      .fillna(value={'Store': s, 'Sales': 0, 'Customers': 0, 'Open': 0,
                                     'Promo': 0, 'StateHoliday': '0',
                                     'SchoolHoliday': 0}))
            filled['DayOfWeek'] = filled.index.weekday + 1
            logger.debug("Store {s} shape after filling {shape}"
                         .format(s=s, shape=filled.shape))
            return filled.reset_index()

        #  DayOfWeek: Monday is 0, Sunday is 6
        logger.info("Expand index of each store to cover full date range")
        train_full = train_csv.groupby('Store').apply(fill_gaps_by_store).reset_index(drop=True)
        assert check_nulls(train_full)
        logger.info("Expanded train data from shape {0} to {1}".format(train_csv.shape, train_full.shape))
        del train_csv
        # gc.collect()

        ##
        test_csv = pd.read_csv(test_file, dtype={'StateHoliday': 'str'}, parse_dates=['Date'])
        logger.info("Read test data: {0} rows, {1} columns".format(*test_csv.shape))

        ##
        test_csv['Open'].fillna(value=1, inplace=True)
        assert check_nulls(test_csv, 'Open')
        logger.debug("Fill nas test_csv Open")

        ##
        # Generate DatetimeIndex for the range of dates in the testing set
        # and the full range over both training and test sets.

        test_range_lims = (test_csv['Date'].min(), test_csv['Date'].max())
        test_range = pd.date_range(test_range_lims[0], test_range_lims[1], name='Date')
        logger.info("Test data range is {0} - {1}".format(test_range.min(), test_range.max()))

        full_range = pd.date_range(min(train_range_lims[0], test_range_lims[0]),
                                   max(train_range_lims[1], test_range_lims[1]), name='Date')
        logger.info("Full data range is {0} - {1}".format(full_range.min(), full_range.max()))

        ##
        fourier_features = fourier(len(full_range), period=365, terms=fourier_terms)

        fourier_features.columns = fourier_names
        fourier_features['Date'] = full_range

        ##
        past_date = pd.to_datetime("2000-01-01")
        future_date = pd.to_datetime("2099-01-01")

        ##
        # Combine data
        tftc = pd.concat([train_full, test_csv])
        logger.info(
            "Combined train and test data. Train data shape {0}. Test data shape {1}. Combined data shape {2}".format(
                train_full.shape, test_csv.shape, tftc.shape
            ))
        del test_csv

        joined = pd.merge(tftc, store, on='Store', how='inner')
        logger.info("Merged with store data, shape {0}".format(joined.shape))

        del store, tftc
        # gc.collect()

        ##
        # Add binary feature for each day of week.
        # ToDo: add term for Sunday.
        for i in range(1, 7):
            joined['DayOfWeek{i}'.format(i=i)] = (joined['DayOfWeek'] == i).astype(int)
            assert check_nulls(joined, 'DayOfWeek{i}'.format(i=i))
        logger.info("Generated Day of Week features")

        ##
        # Add binary features for StateHoliday categories, and one numerical feature
        for i in 'ABC':
            joined['StateHoliday{i}'.format(i=i)] = (joined['StateHoliday'] == i.lower()).astype(int)
            assert check_nulls(joined, 'StateHoliday{i}'.format(i=i))

        letter_to_number = {'0': 1, 'a': 2, 'b': 3, 'c': 4}

        joined['StateHolidayN'] = joined['StateHoliday'].replace(letter_to_number)
        assert check_nulls(joined, 'StateHolidayN')
        logger.info("Generated State Holiday features")

        ##
        joined['CompetitionOpen'] = (joined['Date']
                                     >= joined['CompetitionOpenDate']).astype(int)
        assert check_nulls(joined, 'CompetitionOpen')
        logger.info("Generated Competition Open")

        ##
        letter_to_number = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

        joined['StoreTypeN'] = joined['StoreType'].replace(letter_to_number)
        joined['AssortmentN'] = joined['Assortment'].replace(letter_to_number)
        assert check_nulls(joined, 'StoreTypeN')
        assert check_nulls(joined, 'AssortmentN')
        logger.info("Generated Store Type")

        ##
        # Represent date offsets
        joined['DateTrend'] = ((joined['Date'] - joined['Date'].min())
                               .apply(lambda s: s.days + 1))
        assert check_nulls(joined, 'DateTrend')
        logger.info("Generated Date Trend")

        ##
        # Binary feature to indicate if Promo2 is active
        joined['Promo2Active'] = joined[['Promo2', 'Promo2SinceYear',
                                         'Promo2SinceWeek', 'PromoInterval',
                                         'Date']].apply(is_promo2_active, axis=1)
        assert check_nulls(joined, 'Promo2Active')

        ##
        # Numerical feature for PromoInterval
        # month_to_nums = {'Jan,Apr,Jul,Oct': 3,
        #                  'Feb,May,Aug,Nov': 2,
        #                  'Mar,Jun,Sept,Dec': 4}

        joined['PromoIntervalN'] = (joined['PromoInterval']
                                    .apply(lambda s: 1 if pd.isnull(s) else min(s) + 1))
        assert check_nulls(joined, 'PromoIntervalN')
        logger.info("Generated Promo2 features")

        ##
        # Day, month, year
        joined['MDay'] = joined['Date'].dt.day
        joined['Month'] = joined['Date'].dt.month
        joined['Year'] = joined['Date'].dt.year

        ##
        # Binary feature for day
        for i in range(1, 32):
            joined['MDay{i}'.format(i=i)] = (joined['MDay'] == i).astype(int)

        ##
        # Binary feature for month
        for i in range(1, 13):
            joined['Month{i}'.format(i=i)] = (joined['Month'] == i).astype(int)

        logger.info("Generated date related feature")

        logger.info("Generated direct features, new shape {0}".format(joined.shape))

        ##
        # Apply transformations grouped by Store

        def apply_grouped_by_store(g):
            s = g['Store'].iloc[0]
            logger.debug("Store {s} initial shape {shape}".format(s=s, shape=g.shape))
            g = date_features(g)
            logger.debug("Store {s} after date features shape {shape}".format(s=s, shape=g.shape))
            g = merge_with_fourier_features(g)
            logger.debug("Store {s} final shape {shape}".format(s=s, shape=g.shape))
            return g

        ##
        # Merge fourier features
        def merge_with_fourier_features(g):
            return pd.merge(g, fourier_features, on='Date')

        ##
        def date_features(g):
            g['PromoStarted'] = (g['Promo'].diff() > 0)
            g['Promo2Started'] = (g['Promo2Active'].diff() > 0)
            g['Opened'] = (g['Open'].diff() > 0)
            g['Closed'] = (g['Open'].diff() < 0)
            g['TomorrowClosed'] = g['Closed'].shift(-1).fillna(False)

            # These are incomplete, NA values filled later.
            g['PromoStartedLastDate'] = g.loc[g['PromoStarted'], 'Date']
            g['Promo2StartedDate'] = g.loc[g['Promo2Started'], 'Date']
            g['StateHolidayCLastDate'] = g.loc[g['StateHoliday'] == "c", 'Date']
            g['StateHolidayCNextDate'] = g['StateHolidayCLastDate']
            g['StateHolidayBLastDate'] = g.loc[g['StateHoliday'] == "b", 'Date']
            g['StateHolidayBNextDate'] = g['StateHolidayBLastDate']
            g['StateHolidayANextDate'] = g.loc[g['StateHoliday'] == "a", 'Date']
            g['ClosedLastDate'] = g.loc[g['Closed'], 'Date']
            g['ClosedNextDate'] = g['ClosedLastDate']
            g['OpenedLastDate'] = g.loc[g['Opened'], 'Date']
            g['OpenedNextDate'] = g['OpenedLastDate']
            g['LastClosedSundayDate'] = g.loc[(g['Open'] == 0) & (g['DayOfWeek'] == 7), 'Date']

            # Last dates filled with pad
            features = ['PromoStartedLastDate',
                        'Promo2StartedDate',
                        'StateHolidayCLastDate',
                        'StateHolidayBLastDate',
                        'ClosedLastDate',
                        'OpenedLastDate',
                        'LastClosedSundayDate'
                        ]
            g[features] = (g[features].fillna(method='pad')
                           .fillna(value=pd.Timestamp('1970-01-01 00:00:00')))
            assert check_nulls(g, features)

            # ToDo: check interpretation
            g['IsClosedForDays'] = (g['Date'] - g['ClosedLastDate']).dt.days

            g['LongOpenLastDate'] = (g.loc[(g['Opened'])
                                           & (g['IsClosedForDays'] > 5)
                                           & (g['IsClosedForDays'] < 180),
                                           'Date'])

            # Last dates filled with pad
            features = ['LongOpenLastDate',
                        ]
            g[features] = (g[features].fillna(method='pad')
                           .fillna(value=pd.Timestamp('1970-01-01 00:00:00')))
            assert check_nulls(g, features)

            #
            g.loc[(g['Open'] == 0) & (g['DayOfWeek'] == 7), 'WasClosedOnSunday'] = 1
            g['WasClosedOnSunday'] = (g['WasClosedOnSunday']
                                      .fillna(method='pad', limit=6)
                                      .fillna(value=0))
            assert check_nulls(g, 'WasClosedOnSunday')

            # Next dates filled with backfill
            features = ['StateHolidayCNextDate',
                        'OpenedNextDate',
                        'ClosedNextDate',
                        'StateHolidayBNextDate',
                        'StateHolidayANextDate'
                        ]
            g[features] = (g[features].fillna(method='backfill')
                           .fillna(value=pd.Timestamp('2020-01-01 00:00:00')))
            assert check_nulls(g, features)

            # ToDo: check interpretation
            g['WillBeClosedForDays'] = (g['OpenedNextDate'] - g['Date']).dt.days

            g['LongClosedNextDate'] = (g.loc[(g['Closed'])
                                             & (g['WillBeClosedForDays'] > 5)
                                             & (g['WillBeClosedForDays'] < 180),
                                             'Date'])

            # Next dates filled with backfill
            features = ['LongClosedNextDate',
                        ]
            g[features] = (g[features].fillna(method='backfill')
                           .fillna(value=pd.Timestamp('2020-01-01 00:00:00')))
            assert check_nulls(g, features)

            return g

        ##
        joined = joined.groupby('Store').apply(apply_grouped_by_store)
        logger.info('Expanded with date and fourier features shape {0}'.format(joined.shape))

        ##
        old_shape = joined.shape
        make_decay_features(joined, promo_after=4, promo2_after=3,
                            holiday_b_before=3, holiday_c_before=15, holiday_c_after=3)
        logger.info("Decay features, new shape {shape}".format(shape=joined.shape))
        assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

        ##
        old_shape = joined.shape
        scale_log_features(joined, *decay_features, 'DateTrend')
        logger.info("Scale log features, new shape {shape}".format(shape=joined.shape))
        assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

        ##
        old_shape = joined.shape
        make_before_stairs(joined, "StateHolidayCNextDate", "StateHolidayBNextDate",
                           "StateHolidayANextDate", "LongClosedNextDate",
                           days=stairs_steps)
        logger.info("Before stairs features, new shape {shape}".format(shape=joined.shape))
        assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

        ##
        old_shape = joined.shape
        make_after_stairs(joined, "PromoStartedLastDate", days=(2, 3, 4))
        make_after_stairs(joined, "StateHolidayCLastDate", "StateHolidayBLastDate",
                          "Promo2StartedDate", "LongOpenLastDate",
                          days=stairs_steps)
        logger.info("After stairs features, new shape {shape}".format(shape=joined.shape))
        assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

        ##
        logger.info("Splitting data into train and test, initial shape {shape}".format(shape=joined.shape))
        d1, d2 = train_range_lims
        cls.train = joined.query('@d1 <= Date <= @d2').drop('Id', axis=1)
        logger.info("Train data shape {shape}".format(shape=cls.train.shape))

        ##
        d1, d2 = test_range_lims
        cls.test = joined.query('@d1 <= Date <= @d2').drop(['Sales', 'Customers'], axis=1)
        logger.info("Test data shape {shape}".format(shape=cls.test.shape))

        ##
        if SMALL_TRAIN:
            cls.small_train = cls.train[cls.train['Store'].apply(lambda s: s in example_stores)]
            logger.info("Small train shape {0}".format(cls.small_train.shape))

            cls.small_fold = make_fold(cls.small_train)
            logger.info("Small fold shapes, train: {0}, test:{1}, actual:{2}"
                        .format(cls.small_fold.train.shape, cls.small_fold.test.shape, cls.small_fold.actual.shape))

            cls.one_train = cls.train[cls.train['Store'] == 388]
            cls.one_test = cls.test[cls.test['Store'] == 388]

            logger.info("One train shape {0}, one test shape {1}".format(cls.one_train.shape, cls.one_test.shape))


##
example_stores = (
    388,  # most typical by svd of Sales time series
    562,  # second most typical by svd
    851,  # large gap in 2014
    357  # small gap
)


##
def make_fold(train, step, predict_interval, step_by):
    cross_val_fold = namedtuple('cross_val_fold', 'train_idx test_idx')
    train = train.reset_index(drop=True)
    dates = pd.date_range(train['Date'].min(), train['Date'].max())
    total = dates.shape[0]
    last_train = total - predict_interval - (step - 1) * step_by
    last_train_date = dates[last_train - 1]
    last_predict = last_train + predict_interval
    last_predict_date = dates[last_predict - 1]
    train_idx = train.index[train['Date'] <= last_train_date]
    test_idx = train.index[(train['Date'] > last_train_date) & (train['Date'] <=
                                                                last_predict_date)]
    return cross_val_fold(train_idx=train_idx, test_idx=test_idx)


##
if __name__ == '__main__':
    ##
    store_file, train_file, test_file = "../input/store.csv", "../input/train.csv", "../input/test.csv"
    ##
    Data.process_input(store_file, train_file, test_file)
