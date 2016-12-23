import numpy as np


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


def log_transform_train(train):
    train = train.query('Sales > 0')
    train['Sales'] = np.log(train['Sales'])
    return train
