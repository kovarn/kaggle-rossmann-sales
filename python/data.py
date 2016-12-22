##
from collections import OrderedDict
from itertools import product, chain, starmap

import pandas as pd
import numpy as np

pd.set_option('float_format', "{0:.2f}".format)
##
# md
"""
Read the store data
"""

##
store = pd.read_csv("../input/store.csv")


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


store['CompetitionOpenDate'] = store.apply(get_date, axis=1)

##
train_csv = pd.read_csv("../input/train.csv", low_memory=False)

##
train_csv['Open'] = train_csv['Sales'].apply(lambda s: 1 if s > 0 else 0)
train_csv['Date'] = pd.to_datetime(train_csv['Date'])

##
# Generate DatetimeIndex for the range of dates in the training set
train_range = pd.date_range(train_csv['Date'].min(), train_csv['Date'].max(), name='Date')


##
# Fill in gaps in dates for each store

def fill_gaps_by_store(g):
    s = g['Store'].iloc[0]
    filled = (g.set_index('Date')
              .reindex(train_range)
              .fillna(value={'Store': s, 'Sales': 0, 'Customers': 0, 'Open': 0,
                             'Promo': 0, 'StateHoliday': '0',
                             'SchoolHoliday': 0}))
    filled['DayOfWeek'] = filled.index.weekday + 1
    return filled.reset_index()


#  DayOfWeek: Monday is 0, Sunday is 6
train_full = train_csv.groupby('Store').apply(fill_gaps_by_store).reset_index(drop=True)

##
test_csv = pd.read_csv("../input/test.csv", low_memory=False)

##
test_csv['Open'].fillna(value=1, inplace=True)
test_csv['Date'] = pd.to_datetime(test_csv['Date'])

##
# Generate DatetimeIndex for the range of dates in the testing set
# and the full range over both training and test sets.

test_range = pd.date_range(test_csv['Date'].min(), test_csv['Date'].max(), name='Date')

full_range = pd.date_range(min(train_csv['Date'].min(), test_csv['Date'].min()),
                           max(train_csv['Date'].max(), test_csv['Date'].max()), name='Date')


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
    return A


##
fourier_terms = 5

fourier_features = fourier(len(full_range), period=365, terms=fourier_terms)
fourier_names = ['Fourier' + str(x) for x in range(1, 2 * fourier_terms + 1)]
fourier_features.columns = fourier_names
fourier_features['Date'] = full_range

##
base_linear_features = ["Promo", "Promo2Active", "SchoolHoliday",
                        "DayOfWeek1", "DayOfWeek2", "DayOfWeek2", "DayOfWeek3",
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

stairs_features = chain(
    ["Opened", "PromoStarted", "Promo2Started", "TomorrowClosed",
     "WasClosedOnSunday"],
    map("PromoStartedLastDate{0}after".format, (2, 3, 4)),
    starmap("{1}{0}before".format,
            product(stairs_steps, ("StateHolidayCNextDate", "StateHolidayBNextDate",
                                   "StateHolidayANextDate", "LongClosedNextDate"))),
    starmap("{1}{0}after".format,
            product(stairs_steps, ("StateHolidayCLastDate", "StateHolidayBLastDate",
                                   "Promo2StartedDate", "LongOpenLastDate")))
)

##
month_day_features = map("MDay{0}".format, range(1, 32))

month_features = map("Month{0}".format, range(1, 13))

##
linear_features = chain(base_linear_features, trend_features, decay_features,
                        log_decay_features, stairs_features, fourier_names,
                        month_day_features, month_features)

glm_features = ("Promo", "SchoolHoliday",
                "DayOfWeek1", "DayOfWeek2", "DayOfWeek2", "DayOfWeek3",
                "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
                "StateHolidayA", "StateHolidayB", "StateHolidayC",
                "CompetitionOpen", "PromoDecay", "Promo2Decay",
                "DateTrend", "DateTrendLog",
                "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                "Fourier1", "Fourier2", "Fourier3", "Fourier4")

log_lm_features = ("Promo", "Promo2Active", "SchoolHoliday",
                   "DayOfWeek1", "DayOfWeek2", "DayOfWeek2", "DayOfWeek3",
                   "DayOfWeek4", "DayOfWeek5", "DayOfWeek6",
                   "StateHolidayA", "StateHolidayB", "StateHolidayC",
                   "CompetitionOpen", "PromoDecay", "Promo2Decay",
                   "DateTrend", "DateTrendLog",
                   "StateHolidayCLastDecay", "StateHolidayCNextDecay",
                   "Fourier1", "Fourier2", "Fourier3", "Fourier4")

categorical_numeric_features = chain(
    ("Store", "Promo", "Promo2Active", "SchoolHoliday", "DayOfWeek",
     "StateHolidayN", "CompetitionOpen", "StoreTypeN", "AssortmentN",
     "DateTrend", "MDay", "Month", "Year"), decay_features, stairs_features,
    fourier_names
)

##
past_date = pd.to_datetime("2000-01-01")
future_date = pd.to_datetime("2099-01-01")

##
# Combine data
tftc = pd.concat([train_full, test_csv])
joined = pd.merge(tftc, store, on='Store', how='inner')

##
# Add binary feature for each day of week.
# ToDo: add term for Sunday.
for i in range(1, 7):
    joined['DayOfWeek{i}'.format(i=i)] = (joined['DayOfWeek']
                                          .apply(lambda s: 1 if s == i else 0))

##
# Add binary features for StateHoliday categories, and one numerical feature
for i in 'ABC':
    joined['StateHoliday{i}'.format(i=i)] = (joined['StateHoliday']
                                             .apply(lambda s: 1 if s == i.lower() else 0))

letter_to_number = {'0': 1, 'a': 2, 'b': 3, 'c': 4}

joined['StateHolidayN'] = joined['StateHoliday'].apply(lambda x: letter_to_number[x])

##
joined['CompetitionOpen'] = (joined['Date']
                             >= joined['CompetitionOpenDate']).astype(int)

##
letter_to_number = {'a': 1, 'b': 2, 'c': 3, 'd': 4}

joined['StoreTypeN'] = joined['StoreType'].apply(lambda x: letter_to_number[x])
joined['AssortmentN'] = joined['Assortment'].apply(lambda x: letter_to_number[x])

##
# Represent date offsets
joined['DateTrend'] = ((joined['Date'] - joined['Date'].min())
                       .apply(lambda s: s.days + 1))

##
# Binary feature to indicate if Promo2 is active
# ToDo: Check interpretation
month_to_nums = {'Jan,Apr,Jul,Oct': [1, 4, 7, 10],
                 'Feb,May,Aug,Nov': [2, 5, 8, 11],
                 'Mar,Jun,Sept,Dec': [3, 6, 9, 12]}


def is_promo2_active(row):
    if row['Promo2']:
        start_year = row['Promo2SinceYear']
        start_week = row['Promo2SinceWeek']
        interval = row['PromoInterval']
        date = row['Date']
        if ((date.year > start_year)
            or ((date.year == start_year) and (date.week >= start_week))):
            if date.month in month_to_nums[interval]:
                return 1
    return 0


##
# Numerical feature for PromoInterval
month_to_nums = {'Jan,Apr,Jul,Oct': 3,
                 'Feb,May,Aug,Nov': 2,
                 'Mar,Jun,Sept,Dec': 4}

joined['PromoIntervalN'] = (joined['PromoInterval']
                            .apply(lambda s: 1 if pd.isnull(s)
else month_to_nums[s]))
##
# Day, month, year
joined['MDay'] = joined['Date'].apply(lambda s: s.day)
joined['Month'] = joined['Date'].apply(lambda s: s.month)
joined['Year'] = joined['Date'].apply(lambda s: s.year)

##
# Binary feature for day
for i in range(1, 32):
    joined['MDay{i}'.format(i=i)] = (joined['MDay']
                                     .apply(lambda s: 1 if s == i else 0))

##
# Binary feature for month
for i in range(1, 13):
    joined['Month{i}'.format(i=i)] = (joined['Month']
                                      .apply(lambda s: 1 if s == i else 0))


##
# Apply transformations grouped by Store

def apply_grouped_by_store(g):
    g = date_features(g)
    g = merge_with_fourier_features(g)
    return g


joined = joined.groupby('Store').apply(apply_grouped_by_store)


##
# Merge fourier features
def merge_with_fourier_features(g):
    return pd.merge(g, fourier_features, on='Date')


##
def date_features(g):
    g['PromoStarted'] = g['Promo'].diff().apply(lambda s: 1 if s > 0 else 0)
    g['Promo2Started'] = g['Promo2Active'].diff().apply(lambda s: 1 if s > 0 else 0)
    g['Opened'] = g['Open'].diff().apply(lambda s: 1 if s > 0 else 0)
    g['Closed'] = g['Open'].diff().apply(lambda s: 1 if s < 0 else 0)
    g['TomorrowClosed'] = g['Closed'].shift(-1).fillna(0)

    # These are incomplete, NA values filled later.
    g['PromoStartedLastDate'] = g.loc[g['PromoStarted'] == 1, 'Date']
    g['Promo2StartedDate'] = g.loc[g['Promo2Started'] == 1, 'Date']
    g['StateHolidayCLastDate'] = g.loc[g['StateHoliday'] == "c", 'Date']
    g['StateHolidayCNextDate'] = g['StateHolidayCLastDate']
    g['StateHolidayBLastDate'] = g.loc[g['StateHoliday'] == "b", 'Date']
    g['StateHolidayBNextDate'] = g['StateHolidayBLastDate']
    g['StateHolidayANextDate'] = g.loc[g['StateHoliday'] == "a", 'Date']
    g['ClosedLastDate'] = g.loc[g['Closed'] == 1, 'Date']
    g['ClosedNextDate'] = g['ClosedLastDate']
    g['OpenedLastDate'] = g.loc[g['Opened'] == 1, 'Date']
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
    g[features].fillna(method='pad', inplace=True)
    g[features].fillna(value=pd.Timestamp('1970-01-01 00:00:00'), inplace=True)

    # ToDo: check interpretation
    g['IsClosedForDays'] = (g['Date'] - g['ClosedLastDate']).dt.days

    g['LongOpenLastDate'] = (g.loc[(g['Opened'] == 1)
                                   & (g['IsClosedForDays'] > 5)
                                   & (g['IsClosedForDays'] < 180),
                                   'Date'])

    # Last dates filled with pad
    features = ['LongOpenLastDate',
                ]
    g[features].fillna(method='pad', inplace=True)
    g[features].fillna(value=pd.Timestamp('1970-01-01 00:00:00'), inplace=True)

    #
    g.loc[(g['Open'] == 0) & (g['DayOfWeek'] == 7), 'WasClosedOnSunday'] = 1
    g['WasClosedOnSunday'].fillna(method='pad', limit=6)

    # Next dates filled with backfill
    features = ['StateHolidayCNextDate',
                'OpenedNextDate',
                'ClosedNextDate',
                'StateHolidayBNextDate',
                'StateHolidayANextDate'
                ]
    g[features].fillna(method='backfill', inplace=True)
    g[features].fillna(value=pd.Timestamp('2020-01-01 00:00:00'), inplace=True)

    # ToDo: check interpretation
    g['WillBeClosedForDays'] = (g['OpenedNextDate'] - g['Date']).dt.days

    g['LongClosedNextDate'] = (g.loc[(g['Closed'] == 1)
                                     & (g['WillBeClosedForDays'] > 5)
                                     & (g['WillBeClosedForDays'] < 180),
                                     'Date'])

    # Next dates filled with backfill
    features = ['LongClosedNextDate',
                ]
    g[features].fillna(method='backfill', inplace=True)
    g[features].fillna(value=pd.Timestamp('2020-01-01 00:00:00'), inplace=True)

    return g