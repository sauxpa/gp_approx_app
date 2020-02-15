# gp_approx_app

Bokeh-powered web app to visualize approximation of a continuous function (here absolute value) over a compact interval by a sequence of Gaussian functions, following the representer theorem/GP regression formula.

To run the app :
* clone the repo
* run bokeh serve --show gp_approx_app/

Have fun!


<img src="./spiky_approx.png"
     alt="1e-3 noise, 1e-1.4 scale, 16 points"
     style="float: left; margin-right: 10px;" />
     
<img src="./smooth_approx.png"
     alt="1e-3 noise, 1e-1.4 scale, 32 points"
     style="float: left; margin-right: 10px;" />
