import numpy as np
import pandas as pd
from gp_approximation import Kernel
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource


def make_dataset(f,
                 kernels_to_plot,
                 n=16,
                 scale=1.0,
                 noise=0.1,
                 div_=None,
                 xstart=-1.0,
                 xend=1.0,
                 ):
    """Creates a ColumnDataSource object with data to plot.
    """
    xx = np.linspace(xstart, xend, int(1e3))
    df = pd.DataFrame({'plot_points': xx, 'f': f(xx)}).set_index('plot_points')

    data = np.linspace(xstart, xend, n)
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
    div_.text = err_text

    # Convert dataframe to column data source
    return ColumnDataSource(df)


def make_plot(src, kernel_selection_):
    """Create a figure object to host the plot.
    """
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

    kernels_to_plot = [kernel_selection_.labels[i] for i in kernel_selection_.active]
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
