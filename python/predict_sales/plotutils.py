import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.core.display import display
from bokeh.io import push_notebook, output_notebook
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show

from predict_sales.functions import DataFromHDF

output_notebook()

data = pd.HDFStore('../output/full_run/data.h5')
output = pd.HDFStore('../output/tmp/output.h5')


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

glyphs['Sales'] = p.line('Date', 'Sales', color='navy', legend='actual', source=source)
glyphs['PredictedSales_glm'] = p.line('Date', 'PredictedSales_glm', color='orange', legend='predicted: glm',
                                      source=source)
glyphs['PredictedSales_xgb'] = p.line('Date', 'PredictedSales_xgb', color='red', legend='predicted: xgb', source=source)
glyphs['PredictedSales'] = p.line('Date', 'PredictedSales', color='purple', legend='predicted: mix', source=source)

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
    q.circle('Date', 'RelativeError_glm', color='orange', size=4, alpha=0.9, legend='error: glm', source=esource),
    q.line('Date', 'RelativeError_glm', color='orange', alpha=0.7, legend='error: glm', source=esource))
glyphs['RelativeError_xgb'] = (
    q.circle('Date', 'RelativeError_xgb', color='red', size=4, alpha=0.9, legend='error: xgb', source=esource),
    q.line('Date', 'RelativeError_xgb', color='red', alpha=0.7, legend='error: xgb', source=esource))
glyphs['RelativeError'] = (
    q.circle('Date', 'RelativeError', color='purple', size=4, alpha=0.9, legend='error: mix', source=esource),
    q.line('Date', 'RelativeError', color='purple', alpha=0.7, legend='error: mix', source=esource))

q.title.text = "Prediction errors"
q.legend.location = "top_left"
q.grid.grid_line_alpha = 0
q.xaxis.axis_label = 'Date'
q.yaxis.axis_label = 'Relative Error'


##
def update(store, date_start, date_end, show_zeros):
    store = int(store)
    if store not in stores:
        return

    if show_zeros:
        query = 'Store == {}'.format(store)
    else:
        query = 'Store == {} and Sales > 0'.format(store)

    plot_sales_source = Sales()
    plot_sales_source.actual = sales_source.actual.subset(query)
    plot_sales_source.predict_glm = sales_source.predict_glm.subset(plot_sales_source.actual)
    plot_sales_source.predict_xgb = sales_source.predict_xgb.subset(plot_sales_source.actual)

    plot_sales = (pd.merge(plot_sales_source.predict_glm.get(),
                           plot_sales_source.predict_xgb.get(),
                           on='Id', suffixes=['_glm', '_xgb'])
                  .join(plot_sales_source.actual.get(), on='Id'))

    assert plot_sales.shape[0] == plot_sales_source.actual.select_idx.shape[0]

    plot_sales['PredictedSales'] = (0.97 * plot_sales['PredictedSales_glm'] + 0.985 * plot_sales[
        'PredictedSales_xgb']) / 2

    plot_sales['RelativeError'] = abs(plot_sales['PredictedSales'] - plot_sales['Sales']) / plot_sales['Sales']
    plot_sales['RelativeError_glm'] = abs(plot_sales['PredictedSales_glm'] - plot_sales['Sales']) / plot_sales['Sales']
    plot_sales['RelativeError_xgb'] = abs(plot_sales['PredictedSales_xgb'] - plot_sales['Sales']) / plot_sales['Sales']

    source.data = ColumnDataSource.from_df(plot_sales[source_columns])

    p.x_range.start = np.datetime64(date_start, 'D')
    p.x_range.end = np.datetime64(date_end, 'D')
    p.title.text = str(np.random.randint(1000))
    push_notebook(handle=pn)

    esource.data = ColumnDataSource.from_df(plot_sales[esource_columns])

    q.x_range.start = np.datetime64(date_start, 'D')
    q.x_range.end = np.datetime64(date_end, 'D')
    q.title.text = str(np.random.randint(1000))
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


def update_store(*args):
    s = int(store_text.value)
    if s < stores.max():
        store_text.value = str(s + 1)


next_store.on_click(update_store)

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
w = widgets.interactive(update, store=store_text, date_start=date_start_text,
                        date_end=date_end_text, show_zeros=widgets.fixed(False))

display(widgets.VBox([widgets.HBox([date_range]),
                      widgets.HBox([next_dates, date_start_text, date_end_text]),
                      widgets.HBox([next_store, store_text])]))
