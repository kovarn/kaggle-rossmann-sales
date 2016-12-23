
from .functions import remove_before_changepoint, log_transform_train

import logging

logger = logging.getLogger(__name__)


def process_train(train, test):
    train = remove_before_changepoint(train)
    train = train.query('Store in {test_set_stores}'
                        .format(test_set_stores=list(test['Store'].unique())))
    train = log_transform_train(train)
