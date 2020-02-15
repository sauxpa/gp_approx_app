import numpy as np
import pandas as pd
from gp_approximation import Kernel

from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import CheckboxGroup, Slider, Tabs
from bokeh.layouts import row, WidgetBox


def make_dataset(f, n=1, scale=1.0, noise=0.1):
    xx = np.linspace(-1, 1, 1e3)
    df = pd.DataFrame({'plot_points': xx, 'f': f(xx)}).set_index('plot_points')

    data = np.linspace(-1, 1, n)
    y = f(data)
    kernel = Kernel(data=data, y=y, scale=scale, noise=noise)

    approx_f = np.zeros(xx.shape)
    for weight, func in zip(kernel.weights(), kernel.basis()):
        y = func(xx)
        approx_f += weight*y
    df['GP'] = approx_f

    # Convert dataframe to column data source
    return ColumnDataSource(df)


def make_plot(src):
    # Blank plot with correct labels
    fig = figure(plot_width = 700,
                 plot_height = 700,
                 title = 'GP Approximation',
                 x_axis_label = 'x',
                 y_axis_label = 'y'
                )

    # original function
    fig.line('plot_points',
             'f',
             source = src,
             color = 'color',
             legend = 'True function',
             line_color = 'red'
            )

    # GP approximation
    fig.line('plot_points',
     'GP',
     source = src,
     color = 'color',
     legend = 'GP approximation',
     line_color = 'blue'
    )

    return fig


def update(attr, old, new):
    # Change n to selected value
    n = n_select.value
    scale = 10**logscale_select.value
    noise = 10**lognoise_select.value

    # Create new ColumnDataSource
    new_src = make_dataset(f, n=n, scale=scale, noise=noise)

    # Update the data on the plot
    src.data.update(new_src.data)

# Slider to select n
n_select = Slider(start=1,
                  end=128,
                  step=1,
                  value=16,
                  title='Number of knots'
                 )

# Slider to select logscale
logscale_select = Slider(start=-3.0,
                      end=1.0,
                      step=0.1,
                      value=-1.0,
                      title='Scale (log)'
                      )

# Slider to select noise
lognoise_select = Slider(start=-3.0,
                         end=1.0,
                         step=0.1,
                         value=-1.0,
                         title='Noise (log)'
                         )

# Update the plot when the value is changed
n_select.on_change('value', update)
logscale_select.on_change('value', update)
lognoise_select.on_change('value', update)

# Function to be approximated
def f(x):
    return np.abs(x)

src = make_dataset(f=f)
fig = make_plot(src)
controls = WidgetBox(n_select, logscale_select, lognoise_select)

# Create a row layout
layout = row(controls, fig)

# Make a tab with the layout
tab = Panel(child=layout, title = 'GP Approximation')
tabs = Tabs(tabs=[tab])

curdoc().add_root(tabs)
