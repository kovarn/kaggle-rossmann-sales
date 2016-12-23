import pickle

from predict_sales.data import linear_features
from .functions import remove_before_changepoint, log_transform_train, remove_outliers_lm, select_features, \
    allow_modifications, log_revert_predicted, exp_rmspe, predict_glmnet

import logging

logger = logging.getLogger(__name__)


##
@allow_modifications(False)
def glm_predictions(train, test):
    ##
    logger.info("Dropping store data before changepoint. Initial shape {0}".format(train.shape))
    train_tr = remove_before_changepoint(train)
    logger.info("Reduced to {0}".format(train_tr.shape))

    ##
    logger.info("Dropping stores not in test set. Initial shape {0}".format(train_tr.shape))
    train_tr = train_tr.query('Store in {test_set_stores}'
                              .format(test_set_stores=list(test['Store'].unique())))
    logger.info("Reduced to {0}".format(train_tr.shape))

    ##
    logger.debug("Log transform on sales data")
    train_tr = log_transform_train(train_tr)

    ##
    train_tr = remove_outliers_lm(train_tr)
    logger.info("Removed outliers, reduced shape {0}".format(train_tr.shape))

    ##
    train_tr = select_features(train_tr, linear_features)
    logger.info("Selected linear features, shape {0}".format(train_tr.shape))

    ##
    logger.info("Test shape {0}".format(test.shape))
    test_tr = select_features(test, linear_features)
    logger.info("Test, selected linear features, shape {0}".format(test_tr.shape))

    pred = predict_glmnet(train_tr, test_tr, exp_rmspe, steps=15, step_by=3).predicted
    pred = log_revert_predicted(pred)

    with open('glmnet.pkl', mode='wb') as pkl_file:
        pickle.dump(pred, pkl_file)
