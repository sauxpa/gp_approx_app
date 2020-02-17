import numpy as np

from bokeh.io import curdoc
from bokeh.models import Panel
from bokeh.models.widgets import CheckboxGroup, Slider, Tabs, Div
from bokeh.layouts import row, WidgetBox

from render import make_dataset, make_plot

def update(attr, old, new):
    """Update ColumnDataSource object.
    """
    # change kernels to plot
    kernels_to_plot = [kernel_selection.labels[i] for i in kernel_selection.active]

    # Change n to selected value
    n = n_select.value
    scale = 10**logscale_select.value
    noise = 10**lognoise_select.value

    # Create new ColumnDataSource
    new_src = make_dataset(f,
                           kernels_to_plot,
                           n=n,
                           scale=scale,
                           noise=noise,
                           div_=div,
                           xstart=-1.0,
                           xend=1.0,
                           )

    # Update the data on the plot
    src.data.update(new_src.data)


def update_bm(attr, old, new):
    """Update ColumnDataSource object.
    """
    # change kernels to plot
    kernels_to_plot = [kernel_selection_bm.labels[i] for i in kernel_selection_bm.active]

    # Change n to selected value
    n = n_select_bm.value
    scale = 10**logscale_select_bm.value
    noise = 10**lognoise_select_bm.value

    # Create new ColumnDataSource
    new_src = make_dataset(BM,
                           kernels_to_plot,
                           n=n,
                           scale=scale,
                           noise=noise,
                           div_=div_bm,
                           xstart=0.0,
                           xend=T,
                           )

    # Update the data on the plot
    src_bm.data.update(new_src.data)

available_kernels = ['gauss', 'exp', 'band', 'sinc']
initial_kernels = ['gauss', 'exp', 'band', 'sinc']

### ABSOLUTE VALUE APPROXIMATION
def f(x):
    return np.abs(x)

# select kernel
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

div = Div(text='<b>Absolute errors:</b><br>', width=200, height=100)

src = make_dataset(f,
                   initial_kernels,
                   n=n_select.value,
                   scale=10**logscale_select.value,
                   noise=10**lognoise_select.value,
                   div_=div,
                   xstart=-1.0,
                   xend=1.0,
                   )
fig = make_plot(src, kernel_selection)

controls = WidgetBox(kernel_selection, n_select, logscale_select, lognoise_select, div)

# Create a row layout
layout = row(controls, fig)

# Make a tab with the layout
tab = Panel(child=layout, title='Absolute value')

### BROWNIAN MOTION APPROXIMATION
T = 1.0
N = int(1e3)
tn = np.linspace(0, T, N)
wn = np.random.randn(N)*np.sqrt(T/N)
W = np.cumsum(wn)

def BM(x):
    return np.interp(x, tn, W)

# select kernel
kernel_selection_bm = CheckboxGroup(labels=available_kernels, active=list(range(len(available_kernels))))

# Slider to select n
n_select_bm = Slider(start=1,
                     end=128,
                     step=1,
                     value=16,
                     title='Number of knots'
                     )

# Slider to select logscale
logscale_select_bm = Slider(start=-3.0,
                            end=1.0,
                            step=0.1,
                            value=-1.0,
                            title='Scale (log)'
                            )

# Slider to select noise
lognoise_select_bm = Slider(start=-3.0,
                            end=1.0,
                            step=0.1,
                            value=-1.0,
                            title='Noise (log)'
                            )

# Update the plot when the value is changed
kernel_selection_bm.on_change('active', update_bm)
n_select_bm.on_change('value', update_bm)
logscale_select_bm.on_change('value', update_bm)
lognoise_select_bm.on_change('value', update_bm)
div_bm = Div(text='<b>Absolute errors:</b><br>', width=200, height=100)

src_bm = make_dataset(BM,
                      initial_kernels,
                      n=n_select_bm.value,
                      scale=10**logscale_select_bm.value,
                      noise=10**lognoise_select_bm.value,
                      # which='bm',
                      div_=div_bm,
                      xstart=0.0,
                      xend=T,
                   )
fig_bm = make_plot(src_bm, kernel_selection)

controls_bm = WidgetBox(kernel_selection_bm, n_select_bm, logscale_select_bm, lognoise_select_bm, div_bm)

# Create a row layout
layout = row(controls_bm, fig_bm)

# Make a tab with the layout
tab_bm = Panel(child=layout, title='Brownian motion')

### ALL TABS TOGETHER
tabs = Tabs(tabs=[tab, tab_bm])

curdoc().add_root(tabs)
