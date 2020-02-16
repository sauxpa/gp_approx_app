import numpy as np
import pandas as pd
from gp_approximation import Kernel

from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Panel
from bokeh.models.widgets import CheckboxGroup, Slider, Tabs, Div
from bokeh.layouts import row, WidgetBox


def make_dataset(f, kernels_to_plot, n=16, scale=1.0, noise=0.1):
    xx = np.linspace(-1, 1, 1e3)
    df = pd.DataFrame({'plot_points': xx, 'f': f(xx)}).set_index('plot_points')

    data = np.linspace(-1, 1, n)
    y = f(data)

    err_text = '<b>Absolute errors:</b><br><ul>'
    err = dict()
    for kernel_name in kernels_to_plot:
        kernel = Kernel(data=data, y=y, scale=scale, noise=noise, kernel_name=kernel_name)

        approx_f = np.zeros(xx.shape)
        for weight, func in zip(kernel.weights(), kernel.basis()):
            fx = func(xx)
            approx_f += weight*fx
        df[kernel_name] = approx_f

        err[kernel_name] = np.max(np.abs(df[kernel_name]-df['f']))

    min_err_kernel = min(err, key = err.get)

    for kernel_name in kernels_to_plot:
        if kernel_name == min_err_kernel:
            # highlight the best kernel in bold
            err_text += '<li><b>{}: {:.4f}</b></li>'.format(kernel_name, err[kernel_name])
        else:
            err_text += '<li>{}: {:.4f}</li>'.format(kernel_name, err[kernel_name])

    err_text += '</ul>'
    div.text = err_text
    # Convert dataframe to column data source
    return ColumnDataSource(df)


def make_plot(src):
    # Blank plot with correct labels
    fig = figure(plot_width=700,
                 plot_height=700,
                 title='GP Approximation',
                 x_axis_label='x',
                 y_axis_label='y'
                )

    # original function
    fig.line('plot_points',
             'f',
             source = src,
             color = 'color',
             legend = 'True function',
             line_color = 'red'
            )

    kernels_to_plot = [kernel_selection.labels[i] for i in kernel_selection.active]
    colors = ['blue', 'orange', 'green', 'purple']
    for i, kernel_name in enumerate(kernels_to_plot):
        # GP approximation
        fig.line('plot_points',
         kernel_name,
         source=src,
         color='color',
         legend = 'GP approximation ({})'.format(kernel_name),
         line_color=colors[i]
        )

    fig.legend.click_policy = 'hide'

    return fig


def update(attr, old, new):
    # change kernels to plot
    kernels_to_plot = [kernel_selection.labels[i] for i in kernel_selection.active]

    # Change n to selected value
    n = n_select.value
    scale = 10**logscale_select.value
    noise = 10**lognoise_select.value

    # Create new ColumnDataSource
    new_src = make_dataset(f, kernels_to_plot, n=n, scale=scale, noise=noise)

    # Update the data on the plot
    src.data.update(new_src.data)


# select kernel
available_kernels = ['gauss', 'exp', 'band', 'sinc']
kernel_selection = CheckboxGroup(labels=available_kernels, active=list(range(len(available_kernels))))

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
kernel_selection.on_change('active', update)
n_select.on_change('value', update)
logscale_select.on_change('value', update)
lognoise_select.on_change('value', update)

# Function to be approximated
def f(x):
    return np.abs(x)

initial_kernels = ['gauss', 'exp', 'band', 'sinc']

div = Div(text='<b>Absolute errors:</b><br>', width=200, height=100)

src = make_dataset(f,
                   initial_kernels,
                   n=n_select.value,
                   scale=10**logscale_select.value,
                   noise=10**lognoise_select.value,
                   )
fig = make_plot(src)

controls = WidgetBox(kernel_selection, n_select, logscale_select, lognoise_select, div)

# Create a row layout
layout = row(controls, fig)

# Make a tab with the layout
tab = Panel(child=layout, title='GP Approximation')
tabs = Tabs(tabs=[tab])

curdoc().add_root(tabs)
