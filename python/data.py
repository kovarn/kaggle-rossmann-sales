##
from collections import OrderedDict
from itertools import product, chain, starmap

import pandas as pd
import numpy as np

pd.set_option('float_format',"{0:.2f}".format)
##
# md
"""
Read the store data
"""

##
store = pd.read_csv("../input/store.csv")


##
def get_date(row):
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
# md
"""
Generate DatetimeIndex for the range of dates in the training set
"""
##
train_range = pd.date_range(train_csv['Date'].min(), train_csv['Date'].max(), name='Date')

##
# md
"""
Fill in gaps in dates for each store
"""


##


def fill_gaps_by_store(s, g):
    filled = (g.set_index('Date')
              .reindex(train_range)
              .fillna(value={'Store': s, 'Sales': 0, 'Customers': 0, 'Open': 0,
                             'Promo': 0, 'StateHoliday': '0',
                             'SchoolHoliday': 0}))
    filled['DayOfWeek'] = filled.index.weekday + 1
    return filled.reset_index()


#  DayOfWeek: Monday is 1, Sunday is 7 (add 1 to what is returned by pd.datetime.weekday)
train_full = pd.concat([fill_gaps_by_store(s, g)
                        for s, g in train_csv.groupby('Store')],
                       ignore_index=True)

##
test_csv = pd.read_csv("../input/test.csv", low_memory=False)

##
test_csv['Open'].fillna(value=1, inplace=True)
test_csv['Date'] = pd.to_datetime(test_csv['Date'])

##
# md
"""
Generate DatetimeIndex for the range of dates in the testing set
and the full range over both training and test sets.
"""
##
test_range = pd.date_range(test_csv['Date'].min(), test_csv['Date'].max(), name='Date')

full_range = pd.date_range(min(train_csv['Date'].min(), test_csv['Date'].min()),
                           max(train_csv['Date'].max(), test_csv['Date'].max()), name='Date')

##
# md
"""
Generate Fourier terms
"""


##
def fourier(ts_length, frequency, terms):
    A = pd.DataFrame()
    fns = OrderedDict([('s', np.sin), ('c', np.cos)])
    terms_range = range(1, terms + 1)
    for term, (fn_name, fn) in product(terms_range, fns.items()):
        A[fn_name + str(term)] = fn(2 * np.pi * term * np.arange(1, ts_length + 1) / frequency)
    return A


##
fourier_terms = 5

fourier_features = fourier(len(full_range), frequency=365, terms=fourier_terms)
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
    starmap("{1}{0}before".format, product(stairs_steps, ("StateHolidayCNextDate", "StateHolidayBNextDate",
                                                          "StateHolidayANextDate", "LongClosedNextDate"))),
    starmap("{1}{0}after".format, product(stairs_steps, ("StateHolidayCLastDate", "StateHolidayBLastDate",
                                                         "Promo2StartedDate", "LongOpenLastDate"))))

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
    fourier_names)

##
past_date = pd.to_datetime("2000-01-01")
future_date = pd.to_datetime("2099-01-01")

##
# md
# Combine data
tftc = pd.concat([train_full, test_csv])