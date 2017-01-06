import datetime
import logging

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.core.display import display
from bokeh import palettes
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, gridplot
from scipy.signal import convolve

from predict_sales.functions import DataFromHDF

logger = logging.getLogger(__name__)
# +-from predict_sales import logger

output_notebook()

data = pd.HDFStore('../output/full_run/data.h5')
output = pd.HDFStore('../output/5_stores_output.h5')


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
                  'left', 'right',
                  'PredictedSales_glm', 'PredictedSales_xgb', 'PredictedSales',
                  'RelativeError_glm', 'RelativeError_xgb', 'RelativeError',
                  'AvgRelativeError'
                  ]

source = ColumnDataSource(pd.DataFrame(columns=source_columns))

stores = sales_source.actual.get_column('Store').astype(int).unique()
dates = sales_source.actual.get_column('Date')

##
# create a new plot with a a datetime axis type
p = figure(width=800, height=600, x_axis_type="datetime",
           x_range=(dates.min(), dates.min() + pd.to_timedelta('90 days')))

glyphs = dict()

c_actual, c_mix, c_glm, c_xgb, c_avg_mix = palettes.Dark2_5

glyphs['Sales'] = (
    # p.circle('Date', 'Sales', color=c_actual, size=8, alpha=0.9, legend='Actual', source=source),
    p.quad(left='left', right='right', bottom=0, top='Sales', color=c_actual, alpha=0.4, legend='Actual',
           source=source)
)
glyphs['PredictedSales_glm'] = (
    p.square('Date', 'PredictedSales_glm', color=c_glm, size=8, alpha=0.9, legend='Predicted: glm', source=source),
    p.quad(left='left', right='right', bottom='Sales', top='PredictedSales_glm', color=c_glm, alpha=0.25,
           legend='Predicted: glm', source=source)
)
glyphs['PredictedSales_xgb'] = (
    p.square('Date', 'PredictedSales_xgb', color=c_xgb, size=8, alpha=0.9, legend='Predicted: xgb', source=source),
    p.quad(left='left', right='right', bottom='Sales', top='PredictedSales_xgb', color=c_xgb, alpha=0.25,
           legend='Predicted: xgb', source=source)
)
glyphs['PredictedSales'] = (
    p.diamond('Date', 'PredictedSales', color=c_mix, size=8, alpha=0.9, legend='Predicted: mix', source=source),
    p.quad(left='left', right='right', bottom='Sales', top='PredictedSales', color=c_mix, alpha=0.25,
           legend='Predicted: mix',
           source=source)
)

p.title.text = "Sales by date"
p.legend.location = "top_left"
p.grid.grid_line_alpha = 0
p.xaxis.axis_label = 'Date'
p.yaxis.axis_label = 'Sales'

##
q = figure(width=800, height=400, x_axis_type="datetime",
           x_range=p.x_range, y_range=(0, .5))

# add renderers
glyphs['RelativeError_glm'] = (
    q.quad(bottom=0, top='RelativeError_glm', left='left', right='right',
           color=c_glm, alpha=0.2, legend='Error: glm', source=source)
)
glyphs['RelativeError_xgb'] = (
    q.quad(bottom=0, top='RelativeError_xgb', left='left', right='right',
           color=c_xgb, alpha=0.2, legend='Error: xgb', source=source)
)
glyphs['RelativeError'] = (
    q.quad(bottom=0, top='RelativeError', left='left', right='right',
           color=c_mix, alpha=0.2, legend='Error: mix', source=source)
)
glyphs['AvgRelativeError'] = (
    q.quad(bottom=0, top='AvgRelativeError', left='left', right='right',
           color=c_avg_mix, alpha=0.2, legend='Smoothed RMSPE', source=source)
)

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

    plot_sales['left'] = plot_sales['Date'] - datetime.timedelta(days=0.5)
    plot_sales['right'] = plot_sales['Date'] + datetime.timedelta(days=0.5)

    plot_sales['PredictedSales'] = (0.97 * plot_sales['PredictedSales_glm'] + 0.985 * plot_sales[
        'PredictedSales_xgb']) / 2

    for suffix in ['', '_glm', '_xgb']:
        plot_sales['RelativeError'+suffix] = (abs(plot_sales['PredictedSales'+suffix] - plot_sales['Sales'])
                                              / plot_sales['Sales'])

    window = 31
    plot_sales['AvgRelativeError'] = np.sqrt(convolve(
        np.square(plot_sales['RelativeError']),
        np.ones(window) / window, mode='same'))

    source.data = ColumnDataSource.from_df(plot_sales[source_columns])

    push_notebook(handle=pn)


##
def update_date_range(date_start, date_end):
    p.x_range.start = np.datetime64(date_start, 'D')
    p.x_range.end = np.datetime64(date_end, 'D')
    # p.title.text = str(np.random.randint(1000))
    # push_notebook(handle=pn)

    q.x_range.start = np.datetime64(date_start, 'D')
    q.x_range.end = np.datetime64(date_end, 'D')
    # q.title.text = str(np.random.randint(1000))
    push_notebook(handle=pn)


def show_hide(selections):
    columns = {'xgb': ('PredictedSales_xgb', 'RelativeError_xgb'),
               'glm': ('PredictedSales_glm', 'RelativeError_glm'),
               'mix': ('PredictedSales', 'RelativeError', 'AvgRelativeError')}
    for key, column_names in columns.items():
        visible = key in selections
        for c in column_names:
            try:
                glyphs[c].visible = visible
            except AttributeError:
                for glyph in glyphs[c]:
                    glyph.visible = visible

    push_notebook(handle=pn)


##
next_store = widgets.Button(
    description='Next store',
    disabled=False,
    button_style='info',  # 'success', 'info', 'warning', 'danger' or ''
    tooltip='Click me',
    icon='check'
)

store_text = widgets.Text(
    value=str(stores.min() - 1),
    placeholder='Type something',
    description='Store:',
    disabled=False
)


def update_store_text(*args):
    s = int(store_text.value)
    if s < stores.max():
        store_text.value = str(s + 1)


next_store.on_click(update_store_text)

date_range = widgets.Dropdown(options=[('30 days', 30), ('90 days', 90), ('180 days', 180), ('365 days', 365)], value=90)

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

selections = widgets.SelectMultiple(options=['xgb', 'glm', 'mix'], value=('xgb', 'glm', 'mix'))

##
# show the results
g = gridplot([[p], [q]])
pn = show(g, notebook_handle=True)

# qn = show(q, notebook_handle=True)

##
w1 = widgets.interactive(update_store, store=store_text, show_zeros=widgets.fixed(False))
w2 = widgets.interactive(update_date_range, date_start=date_start_text, date_end=date_end_text)
w3 = widgets.interactive(show_hide, selections=selections)

display(widgets.VBox([widgets.HBox([date_range, selections]),
                      widgets.HBox([next_dates, date_start_text, date_end_text]),
                      widgets.HBox([next_store, store_text])]))
