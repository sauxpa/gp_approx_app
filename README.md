# gp_approx_app

Bokeh-powered web app to visualize approximation of continuous functions over a compact interval by a sequence of Gaussian functions, following the representer theorem/GP regression formula.

Available examples:
* Absolute value,
* Realization of Brownian motion.

To run the app :
* clone the repo
* run bokeh serve --show gp_approx_app/

Have fun!


### 16 points, increasing scale

<img src="./scale_increase.gif"
     alt="1e-3 noise, 16 points, scale increase"
     style="float: left; margin-right: 10px;" />
     
### 16 points

<img src="./spiky_approx.png"
     alt="1e-3 noise, 1e-1.4 scale, 16 points, Gaussian kernel"
     style="float: left; margin-right: 10px;" />
     

### 32 points

<img src="./smooth_approx.png"
     alt="1e-3 noise, 1e-1.4 scale, 32 points, Gaussian kernel"
     style="float: left; margin-right: 10px;" />
