##
# import pyximport;
#
# pyximport.install()
import warnings

from predict_sales import data_helpers
import logging
from collections import OrderedDict, namedtuple
from functools import partial
from itertools import product, chain, starmap

import numpy as np
import pandas as pd

from tqdm import tqdm


# ToDo: Remove this
REDUCE_DATA = 5
SMALL_TRAIN = False
CYTHON = False

logger = logging.getLogger(__name__)
# from predict_sales import logger
logger.info("Module loaded")

pd.set_option('float_format', "{0:.2f}".format)
pd.set_option('io.hdf.default_format','table')

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


# 'Date', 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'PromoInterval'

def is_promo2_active(date, promo2, start_year, start_week, interval):
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
    g['Promo2Decay'] = (g['Promo2Active'] * np.maximum(
        promo2_after - (g['Date'] - g['Promo2StartedDate']).dt.days
        , 0)).astype(float)
    g['StateHolidayBNextDecay'] = (1 - g['StateHolidayB']) * np.maximum(
        holiday_b_before - (g['StateHolidayBNextDate'] - g['Date']).dt.days
        , 0)
    g['StateHolidayCNextDecay'] = (1 - g['StateHolidayC']) * np.maximum(
        holiday_c_before - (g['StateHolidayCNextDate'] - g['Date']).dt.days
        , 0)

    assert check_nulls(g, 'Promo2Decay')


##
def scale_log_features(g, *features):
    for f in features:
        if g[f].max() > 0:
            g[f] /= g[f].max()
        g["{0}Log".format(f)] = np.log1p(g[f])
        if g["{0}Log".format(f)].max() > 0:
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
    @classmethod
    def process_input(cls, store_file, train_file, test_file, output_file):
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
        train_csv = pd.read_csv(train_file, dtype={'StateHoliday': 'str', 'Store': int}, parse_dates=['Date'])
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
        tqdm.pandas(desc="Expand index of stores")
        train_full = train_csv.groupby('Store').progress_apply(fill_gaps_by_store).reset_index(drop=True)
        assert check_nulls(train_full)
        logger.info("Expanded train data from shape {0} to {1}".format(train_csv.shape, train_full.shape))
        del train_csv
        # gc.collect()

        ##
        test_csv = pd.read_csv(test_file, dtype={'StateHoliday': 'str', 'Store': int}, parse_dates=['Date'])
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
        test_csv.set_index('Id', inplace=True)
        logger.info('Changed index to Id for test set, shape {}'.format(test_csv.shape))

        ##
        train_start = test_csv.index.max() + 1
        train_idx = pd.RangeIndex(train_start, train_start + train_full.shape[0])
        train_full.index = train_idx
        train_full.index.name = 'Id'

        ##
        tftc = pd.concat([train_full, test_csv])  # type: pd.DataFrame
        logger.info(
            "Combined train and test data. Train data shape {0}. Test data shape {1}. Combined data shape {2}".format(
                train_full.shape, test_csv.shape, tftc.shape
            ))
        del test_csv
        assert len(tftc.index) == len(set(tftc.index))
        assert tftc.index.name == 'Id'

        ##
        # Add binary feature for each day of week.
        # ToDo: add term for Sunday.
        for i in range(1, 7):
            tftc['DayOfWeek{i}'.format(i=i)] = (tftc['DayOfWeek'] == i).astype(int)
            assert check_nulls(tftc, 'DayOfWeek{i}'.format(i=i))
        logger.info("Generated Day of Week features")

        ##
        # Add binary features for StateHoliday categories, and one numerical feature
        for i in 'ABC':
            tftc['StateHoliday{i}'.format(i=i)] = (tftc['StateHoliday'] == i.lower()).astype(int)
            assert check_nulls(tftc, 'StateHoliday{i}'.format(i=i))

        letter_to_number = {'0': 1, 'a': 2, 'b': 3, 'c': 4}

        tftc['StateHolidayN'] = tftc['StateHoliday'].replace(letter_to_number)
        assert check_nulls(tftc, 'StateHolidayN')
        logger.info("Generated State Holiday features")

        ##
        letter_to_number = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

        store['StoreTypeN'] = store['StoreType'].replace(letter_to_number)
        store['AssortmentN'] = store['Assortment'].replace(letter_to_number)
        assert check_nulls(store, 'StoreTypeN')
        assert check_nulls(store, 'AssortmentN')
        logger.info("Generated Store Type")

        ##
        # Represent date offsets
        tftc['DateTrend'] = (tftc['Date'] - tftc['Date'].min()).dt.days
        assert check_nulls(tftc, 'DateTrend')
        logger.info("Generated Date Trend")

        ##
        store['PromoIntervalN'] = (store['PromoInterval']
                                   .apply(lambda s: 1 if pd.isnull(s) else min(s) + 1))
        assert check_nulls(store, 'PromoIntervalN')
        logger.info("Generated PromoInterval features")

        ##
        # Day, month, year
        tftc['MDay'] = tftc['Date'].dt.day
        tftc['Month'] = tftc['Date'].dt.month
        tftc['Year'] = tftc['Date'].dt.year

        ##
        # Binary feature for day
        for i in range(1, 32):
            tftc['MDay{i}'.format(i=i)] = (tftc['MDay'] == i).astype(int)

        ##
        # Binary feature for month
        for i in range(1, 13):
            tftc['Month{i}'.format(i=i)] = (tftc['Month'] == i).astype(int)

        logger.info("Generated date related feature")

        logger.info("Generated direct features, new shape {0}".format(tftc.shape))

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
            return pd.merge(g.reset_index(), fourier_features, on='Date', sort=True).set_index('Id')

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
        grouped_by_store = tftc.groupby('Store')
        store.set_index('Store', inplace=True)

        data = pd.HDFStore(output_file, mode='w')

        with warnings.catch_warnings():
            # We are working with DataFrame copy in joined. So ignore this warning.
            warnings.simplefilter(action='ignore', category=pd.core.common.SettingWithCopyWarning)
            for s, joined in tqdm(grouped_by_store, desc="Generating features by store"):
                s = int(s)
                try:
                    store_details = store.loc[s, :]
                except KeyError:
                    if REDUCE_DATA:
                        break
                    else:
                        raise
                joined['CompetitionOpen'] = (joined['Date']
                                             >= store_details['CompetitionOpenDate']).astype(int)
                logger.debug("Store: {: 5d}, Generated Competition Open".format(s))
                assert check_nulls(joined, 'CompetitionOpen')

                # Binary feature to indicate if Promo2 is active
                if not CYTHON:
                    joined['Promo2Active'] = joined['Date'].apply(
                        partial(is_promo2_active, promo2=store_details['Promo2'],
                                start_year=store_details['Promo2SinceYear'],
                                start_week=store_details['Promo2SinceWeek'],
                                interval=store_details['PromoInterval'])
                    )
                    assert check_nulls(joined, 'Promo2Active')
                else:
                    joined['Promo2Active'] = data_helpers.is_promo2_active(
                        store_details['Promo2'], np.int64(store_details['Promo2SinceYear']),
                        np.int64(store_details['Promo2SinceWeek']),
                        joined['Date'].dt.year.values, joined['Date'].dt.week.values, joined['Date'].dt.month.values,
                        store_details['PromoInterval'])
                    assert check_nulls(joined, 'Promo2Active')
                logger.debug("Store: {: 5d}, Generated Promo2 features".format(s))

                joined = merge_with_fourier_features(joined)
                logger.debug("Store: {: 5d}, Fourier features, new shape {shape}".format(s, shape=joined.shape))
                joined = date_features(joined)
                logger.debug("Store: {: 5d}, Date features, new shape {shape}".format(s, shape=joined.shape))

                old_shape = joined.shape
                make_decay_features(joined, promo_after=4, promo2_after=3,
                                    holiday_b_before=3, holiday_c_before=15, holiday_c_after=3)
                logger.debug("Store: {: 5d}, Decay features, new shape {shape}".format(s, shape=joined.shape))
                assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

                old_shape = joined.shape
                scale_log_features(joined, *decay_features, 'DateTrend')
                logger.debug("Store: {: 5d}, Scale log features, new shape {shape}".format(s, shape=joined.shape))
                assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

                old_shape = joined.shape
                make_before_stairs(joined, "StateHolidayCNextDate", "StateHolidayBNextDate",
                                   "StateHolidayANextDate", "LongClosedNextDate",
                                   days=stairs_steps)
                logger.debug("Store: {: 5d}, Before stairs features, new shape {shape}".format(s, shape=joined.shape))
                assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

                old_shape = joined.shape
                make_after_stairs(joined, "PromoStartedLastDate", days=(2, 3, 4))
                make_after_stairs(joined, "StateHolidayCLastDate", "StateHolidayBLastDate",
                                  "Promo2StartedDate", "LongOpenLastDate",
                                  days=stairs_steps)
                logger.debug("Store: {: 5d}, After stairs features, new shape {shape}".format(s, shape=joined.shape))
                assert joined.shape[0] == old_shape[0] and joined.shape[1] > old_shape[1]

                check_nulls(joined, 'Promo2Decay')

                store_train_idx = joined.index.intersection(train_idx)

                store_test_idx = joined.index.difference(train_idx)

                assert len(store_train_idx) + len(store_test_idx) == joined.shape[0]

                train = joined.loc[store_train_idx, :]
                data.append('train', train, data_columns=['Store', 'Date', 'Sales'])
                logger.debug("Store: {: 5d}, Wrote train data, shape {shape}".format(s, shape=train.shape))

                test = joined.loc[store_test_idx, :]
                data.append('test', test, data_columns=['Store', 'Date', 'Sales', 'Open'])
                logger.debug("Store: {: 5d}, Wrote test data, shape {shape}".format(s, shape=test.shape))

        store.drop('PromoInterval', axis=1, inplace=True)
        logger.debug('Removed PromoInterval from store')

        data.put('store', store)
        logger.info('Wrote data to file {}'.format(output_file))
        logger.info(repr(data))
        data.close()

##
example_stores = (
    388,  # most typical by svd of Sales time series
    562,  # second most typical by svd
    851,  # large gap in 2014
    357  # small gap
)


##
def make_fold(train, date, step, predict_interval, step_by):
    cross_val_fold = namedtuple('cross_val_fold', 'train_idx test_idx')
    train = train.reset_index(drop=True)
    dates = pd.date_range(date.min(), date.max())
    total = dates.shape[0]
    last_train = total - predict_interval - (step - 1) * step_by
    last_train_date = dates[last_train - 1]
    last_predict = last_train + predict_interval
    last_predict_date = dates[last_predict - 1]
    train_idx = train.index[date <= last_train_date]
    test_idx = train.index[(date > last_train_date) & (date <= last_predict_date)]
    return cross_val_fold(train_idx=train_idx, test_idx=test_idx)


##
if __name__ == '__main__':
    ##
    store_file, train_file, test_file = "../input/store.csv", "../input/train.csv", "../input/test.csv"
    output_file = "../output/data.h5"
    ##
    Data.process_input(store_file, train_file, test_file, output_file)
