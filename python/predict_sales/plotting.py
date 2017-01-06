import logging

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.core.display import display
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

from predict_sales.functions import DataFromHDF

logger = logging.getLogger(__name__)
# +-from predict_sales import logger

output_notebook()

data = pd.HDFStore('../output/full_run/data.h5')
output = pd.HDFStore('../output/output.h5')


class Sales:
    def __init__(self, actual=None, predict_glm=None, predict_xgb=None):
        self.actual = actual
        self.predict_glm = predict_glm
        self.predict_xgb = predict_xgb


select_idx = range(40)
# prepare some data

sales_source = Sales()
sales_source.actual = DataFromHDF(data_store=data, key='train', columns=['Date', 'Sales'])
sales_source.predict_glm = DataFromHDF(data_store=output, key='train/glm')
sales_source.predict_xgb = DataFromHDF(data_store=output, key='train/xgb')

source_columns = ['Id', 'Sales', 'Date',
                  'PredictedSales_glm', 'PredictedSales_xgb', 'PredictedSales',
                  ]
esource_columns = ['Id', 'Date',
                   'RelativeError_glm', 'RelativeError_xgb', 'RelativeError']

source = ColumnDataSource(pd.DataFrame(columns=source_columns))
esource = ColumnDataSource(pd.DataFrame(columns=esource_columns))

stores = sales_source.actual.get_column('Store').astype(int).unique()
dates = sales_source.actual.get_column('Date')

##
# create a new plot with a a datetime axis type
p = figure(width=800, height=600, x_axis_type="datetime",
           x_range=(dates.min(), dates.min() + pd.to_timedelta('180 days')))

glyphs = dict()

# add renderers
# p.circle('Date', 'Sales', size=8, fill_color='white', alpha=1, source=source)

glyphs['Sales'] = (p.circle('Date', 'Sales', color='navy', size=4, alpha=0.9, legend='actual', source=source),
                   p.line('Date', 'Sales', color='navy', legend='actual', source=source))
glyphs['PredictedSales_glm'] = (p.square('Date', 'PredictedSales_glm', color='orange', size=4, alpha=0.9, legend='predicted: glm',
                                      source=source),
                                p.line('Date', 'PredictedSales_glm', color='orange', alpha=0.7, legend='predicted: glm',
                                      source=source))
glyphs['PredictedSales_xgb'] = (p.square('Date', 'PredictedSales_xgb', color='red', size=4, alpha=0.9, legend='predicted: xgb', source=source),
                                p.line('Date', 'PredictedSales_xgb', color='red', alpha=0.7, legend='predicted: xgb', source=source))
glyphs['PredictedSales'] = (p.diamond('Date', 'PredictedSales', color='purple', size=4, alpha=0.9, legend='predicted: mix', source=source),
                            p.line('Date', 'PredictedSales', color='purple', alpha=0.7, legend='predicted: mix', source=source))

# NEW: customize by setting attributes
p.title.text = "Sales by date"
p.legend.location = "top_left"
p.grid.grid_line_alpha = 0
p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Sales'
p.ygrid.band_fill_color = "olive"
p.ygrid.band_fill_alpha = 0.1

##
q = figure(width=800, height=600, x_axis_type="datetime",
           x_range=(dates.min(), dates.min() + pd.to_timedelta('180 days')))

# add renderers
# p.circle('Date', 'Sales', size=8, fill_color='white', alpha=1, source=source)

glyphs['RelativeError_glm'] = (
    q.square('Date', 'RelativeError_glm', color='orange', size=4, alpha=0.9, legend='error: glm', source=esource),
    q.line('Date', 'RelativeError_glm', color='orange', alpha=0.7, legend='error: glm', source=esource))
glyphs['RelativeError_xgb'] = (
    q.square('Date', 'RelativeError_xgb', color='red', size=4, alpha=0.9, legend='error: xgb', source=esource),
    q.line('Date', 'RelativeError_xgb', color='red', alpha=0.7, legend='error: xgb', source=esource))
glyphs['RelativeError'] = (
    q.diamond('Date', 'RelativeError', color='purple', size=4, alpha=0.9, legend='error: mix', source=esource),
    q.line('Date', 'RelativeError', color='purple', alpha=0.7, legend='error: mix', source=esource))

q.title.text = "Prediction errors"
q.legend.location = "top_left"
q.grid.grid_line_alpha = 0
q.xaxis.axis_label = 'Date'
q.yaxis.axis_label = 'Relative Error'


##
def update_store(store, show_zeros):
    store = int(store)
    if store not in stores:
        return

    if show_zeros:
        query = 'Store == {}'.format(store)
    else:
        query = 'Store == {} and Sales > 0'.format(store)

    plot_sales_source = Sales()
    plot_sales_source.actual = sales_source.actual.subset(query)
    actual_id = plot_sales_source.actual.get_index()
    logger.debug('Store {0} data length: {1}'.format(store, len(actual_id)))
    plot_sales_source.predict_glm = sales_source.predict_glm.subset('Id in {}'.format(list(actual_id)))
    logger.debug('glm predictions length {0}'.format(len(plot_sales_source.predict_glm.select_idx)))
    plot_sales_source.predict_xgb = sales_source.predict_xgb.subset('Id in {}'.format(list(actual_id)))
    logger.debug('xgb predictions length {0}'.format(len(plot_sales_source.predict_xgb.select_idx)))

    plot_sales = (pd.merge(plot_sales_source.predict_glm.get(),
                           plot_sales_source.predict_xgb.get(),
                           on='Id', suffixes=['_glm', '_xgb'])
                  .join(plot_sales_source.actual.get(), on='Id', how='inner'))

    logger.debug('Plot sales for store {0} shape: {1}'.format(store, plot_sales.shape))

    # assert plot_sales.shape[0] == plot_sales_source.actual.select_idx.shape[0]

    # plot_sales['left'] = plot_sales['Date'] - datetime.timedelta(days=0.5)
    # plot_sales['right'] = plot_sales['Date'] + datetime.timedelta(days=0.5)

    plot_sales['PredictedSales'] = (0.97 * plot_sales['PredictedSales_glm'] + 0.985 * plot_sales[
        'PredictedSales_xgb']) / 2

    plot_sales['RelativeError'] = abs(plot_sales['PredictedSales'] - plot_sales['Sales']) / plot_sales['Sales']
    plot_sales['RelativeError_glm'] = abs(plot_sales['PredictedSales_glm'] - plot_sales['Sales']) / plot_sales['Sales']
    plot_sales['RelativeError_xgb'] = abs(plot_sales['PredictedSales_xgb'] - plot_sales['Sales']) / plot_sales['Sales']

    source.data = ColumnDataSource.from_df(plot_sales[source_columns])

    push_notebook(handle=pn)

    esource.data = ColumnDataSource.from_df(plot_sales[esource_columns])

    push_notebook(handle=qn)


##
def update_date_range(date_start, date_end):

    p.x_range.start = np.datetime64(date_start, 'D')
    p.x_range.end = np.datetime64(date_end, 'D')
    # p.title.text = str(np.random.randint(1000))
    push_notebook(handle=pn)

    q.x_range.start = np.datetime64(date_start, 'D')
    q.x_range.end = np.datetime64(date_end, 'D')
    # q.title.text = str(np.random.randint(1000))
    push_notebook(handle=qn)


##
next_store = widgets.Button(
    description='Next store',
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check'
)

store_text = widgets.Text(
    value=str(stores.min()),
    placeholder='Type something',
    description='Store:',
    disabled=False
)


def update_store_text(*args):
    s = int(store_text.value)
    if s < stores.max():
        store_text.value = str(s + 1)


next_store.on_click(update_store_text)

date_range = widgets.Dropdown(options={'30 days': 30, '90 days': 90, '180 days': 180, '365 days': 365}, value=90)

next_dates = widgets.Button(
    description='Next dates',
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check'
)

date_start_text = widgets.Text(
    value=str(np.datetime64(dates.min(), 'D')),
    placeholder='Type something',
    description='Store:',
    disabled=False
)

date_end_text = widgets.Text(
    value=str(np.datetime64(dates.min(), 'D') + np.timedelta64(date_range.value, 'D')),
    placeholder='Type something',
    description='Store:',
    disabled=False
)


def get_next_dates(*args):
    de = date_end_text.value
    date_start_text.value = de
    date_end_text.value = str(np.datetime64(de, 'D') + np.timedelta64(date_range.value, 'D'))


def set_new_range(*args):
    ds = date_start_text.value
    date_end_text.value = str(np.datetime64(ds, 'D') + np.timedelta64(date_range.value, 'D'))


next_dates.on_click(get_next_dates)
date_range.observe(set_new_range, 'value')

##
# show the results
pn = show(p, notebook_handle=True)

qn = show(q, notebook_handle=True)

##
w1 = widgets.interactive(update_store, store=store_text, show_zeros=widgets.fixed(False))
w2 = widgets.interactive(update_date_range, date_start=date_start_text, date_end=date_end_text)

display(widgets.VBox([widgets.HBox([date_range]),
                      widgets.HBox([next_dates, date_start_text, date_end_text]),
                      widgets.HBox([next_store, store_text])]))
