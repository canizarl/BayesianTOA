from bayes_positioner import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

import pymc3 as pm
from scipy.stats import gaussian_kde
import arviz as az
from astropy.constants import c, m_e, R_sun, e, eps0, au
import heliospacecraftlocation as hsl

from math import sqrt
from joblib import Parallel, delayed
#from multiprocessing import Pool


def parallel_pos_map(x1,y1):
    report = 0
    trace_plot = 0
    print(f"P({x1},{y1})")
    x_true = np.array([x1 * R_sun.value, y1 * R_sun.value])  # true source position (m)
    v_true = c.value  # speed of light (m/s)
    t0_true = 100  # source time. can be any constant, as long as it is within the uniform distribution prior on t0
    d_true = np.linalg.norm(stations - x_true, axis=1)
    t1_true = d_true / v_true  # true time of flight values
    # t_obs = t1_true-t0_true# true time difference of arrival values
    t_obs = t1_true
    np.random.seed(1)
    # t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations

    mu, sd, t1_pred, trace, summary = triangulate(stations, t_obs, report=0, plot=0)

    if report == 1:
        # report
        print(summary)
        print(f"t0_true: {t0_true}")
        print(f"t_obs:    {t_obs}")
        print(f"t1_true: {t1_true}")
        print(f"t1_pred: {t1_pred}")
        print(f"stations: {stations}")

    if trace_plot == 1:
        # trace plot
        ax, ay = az.plot_trace(trace, compact=False)[1:3, 0]
        ax.hlines(0.6065 * ax.get_ylim()[1], mu[0] - sd[0], mu[0] + sd[0])  # add mu, sigma lines to x,y plots
        ay.hlines(0.6065 * ay.get_ylim()[1], mu[1] - sd[1], mu[1] + sd[1])
        # plt.savefig("bayes_positioner_result1.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
        # pm.autocorrplot(trace)
        # pm.plot_posterior(trace)

    # difference detected observed
    x_true = x_true / R_sun.value
    xy = mu[0] / R_sun.value, mu[1] / R_sun.value

    buffer = sqrt((xy[0] - x_true[0]) ** 2 + (xy[1] - x_true[1]) ** 2)
    return buffer



if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    print('Running on PyMC3 v{}'.format(pm.__version__))


    # generate some test data
    N_STATIONS = 4
    np.random.seed(1)


    stations = np.array([[-100, -100], [100, -100], [-100, 100],[100,100]])

    stations = stations*R_sun.value

    delta_obs = []

    x1 = 1
    y1 = 1.2
    xrange = [-50,50]
    xres = 1
    yrange = [-50,50]
    yres =1
    xmapaxis = np.arange(xrange[0], xrange[1], xres)
    ymapaxis = np.arange(yrange[0], yrange[1], yres)



    threads = 4
    #delta_obs1 = Parallel(n_jobs=2)(delayed(parallel_pos_map)(x1, y1) for x1 in range(5) for y1 in range(5))

    # for x1 in xmapaxis:
    #     for y1 in ymapaxis:
    #         bufferval = parallel_pos_map(x1, y1)
    #         delta_obs.append(bufferval)

    results = Parallel(n_jobs=1)(delayed(parallel_pos_map)(i, j) for i in range(5) for j in range(2))


    # local map
    stations = stations / R_sun.value





    delta_obs = np.reshape(delta_obs1, (len(xmapaxis), len(ymapaxis)))




from bayes_positioner import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

import os

import pymc3 as pm
from scipy.stats import gaussian_kde
import arviz as az
from astropy.constants import c, m_e, R_sun, e, eps0, au
import heliospacecraftlocation as hsl
import datetime as dt

from math import sqrt, radians


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    print('Running on PyMC3 v{}'.format(pm.__version__))


    # generate some test data
    N_STATIONS = 5
    np.random.seed(1)


    L1 = [0.99*(au/R_sun),0]
    L4 = [(au/R_sun)*np.cos(radians(60)),(au/R_sun)*np.sin(radians(60))]
    L5 = [(au/R_sun)*np.cos(radians(60)),-(au/R_sun)*np.sin(radians(60))]
    ahead = [0, (au/R_sun)]
    behind = [0, -(au/R_sun)]

    stations = np.array([L1, L4, L5, ahead, behind])

    stations = stations*R_sun.value

    delta_obs = []

    #x1 = 1
    #y1 = 1.2
    xrange = [-200,200]
    xres = 10
    yrange = [-200,200]
    yres = xres
    xmapaxis = np.arange(xrange[0], xrange[1], xres)
    ymapaxis = np.arange(yrange[0], yrange[1], yres)

    pixels = ((xrange[1]-xrange[0])/xres) * ((yrange[1]-yrange[0])/yres)

    for x1 in xmapaxis:
        for y1 in ymapaxis:
            if pixels == (((xrange[1] - xrange[0]) / xres) * ((yrange[1] - yrange[0]) / yres)):
                print("-----------------------------------")
                print(f"Time estimate: N/A")
                print(f"Pixels left:{pixels}")
                print(f"Current Pixel: {x1, y1}")
                print(f"Last loop time: N/A")
                print("-----------------------------------")

            else:
                timeestimate = timeloop * pixels
                print("-----------------------------------")
                print(f"Time estimate: {timeestimate}")
                print(f"Pixels left:{pixels}")
                print(f"Current Pixel: {x1, y1}")
                print(f"Last loop time: {timeloop}")
                print("-----------------------------------")
            timeloopstart = dt.datetime.now()
            try:
                x_true = np.array([x1 * R_sun.value, y1 * R_sun.value])  # true source position (m)
                v_true = c.value  # speed of light (m/s)
                t0_true = 100  # source time. can be any constant, as long as it is within the uniform distribution prior on t0
                d_true = np.linalg.norm(stations - x_true, axis=1)
                t1_true = d_true / v_true  # true time of flight values
                # t_obs = t1_true-t0_true# true time difference of arrival values
                t_obs = t1_true
                np.random.seed(1)
                # t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations

                mu, sd, t1_pred, trace, summary = triangulate(stations, t_obs, report=0, plot=0)

                # report
                print(summary)
                print(f"t0_true: {t0_true}")
                print(f"t_obs:    {t_obs}")
                print(f"t1_true: {t1_true}")
                print(f"t1_pred: {t1_pred}")
                print(f"stations: {stations}")


                traceplotpath = f"./traceplot_{xrange[0]}_{xrange[1]}_{yrange[0]}_{yrange[1]}_{xres}_{yres}"
                isExist = os.path.exists(traceplotpath)

                if not isExist:
                    # Create a new directory because it does not exist
                    os.makedirs(traceplotpath)
                    print("The new directory is created!")


                # trace plot
                ax, ay = az.plot_trace(trace, compact=False)[1:3, 0]
                ax.hlines(0.6065 * ax.get_ylim()[1], mu[0] - sd[0], mu[0] + sd[0])  # add mu, sigma lines to x,y plots
                ay.hlines(0.6065 * ay.get_ylim()[1], mu[1] - sd[1], mu[1] + sd[1])
                plt.savefig(f"./traceplotpath/bayes_positioner_traceplot_{x1}_{y1}.jpg", bbox_inches='tight', pad_inches=0.01, dpi=100)
                plt.close()
                # pm.autocorrplot(trace)
                # pm.plot_posterior(trace)


                # difference detected observed
                x_true = x_true / R_sun.value
                xy = mu[0] / R_sun.value, mu[1] / R_sun.value

                delta_obs.append(sqrt((xy[0] - x_true[0]) ** 2 + (xy[1] - x_true[1]) ** 2))
                pixels = pixels - 1


            except:
                delta_obs.append(100)
                pixels = pixels - 1
                pass
            timeloop = dt.datetime.now() - timeloopstart

    # local map
    stations = stations / R_sun.value





    delta_obs = np.reshape(delta_obs, (len(xmapaxis), len(ymapaxis)))



    plt.figure(figsize=(8,8))
    plt.scatter(stations[:,0], stations[:,1], marker="^", s=80, label="Receivers")
    #plt.scatter(x_true[0], x_true[1], s=40, label="True source position")
    plt.pcolormesh(xmapaxis, ymapaxis, delta_obs, cmap='RdBu', vmin=np.min(delta_obs), vmax=np.max(delta_obs))
    plt.colorbar()
    #ell = matplotlib.patches.Ellipse(xy=(mu[0]/R_sun.value, mu[1]/R_sun.value),
    #          width=4*sd[0]/R_sun.value, height=4*sd[1]/R_sun.value,
    #          angle=0., color='black', label="Posterior ($2\sigma$)", lw=1.5)
    #plt.plot(mu[0]/R_sun.value, mu[1]/R_sun.value,'k*')
    #ell.set_facecolor('none')
    #plt.gca().add_patch(ell)
    #plt.legend(loc=1)
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    plt.plot(au/R_sun,0,'bo')
    plt.plot(0,0, 'yo')
    plt.xlabel("'HEE - X / $R_{\odot}$'")
    plt.ylabel("'HEE - Y / $R_{\odot}$'")
    #plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.show(block=False)



    plt.figure(figsize=(8,8))
    plt.axis('equal')

    plt.scatter(stations[:,0], stations[:,1], marker="^", s=80, label="Receivers")
    plt.plot(au/R_sun,0,'bo')
    plt.plot(0,0, 'yo')
    plt.xlabel("'HEE - X / $R_{\odot}$'")
    plt.ylabel("'HEE - Y / $R_{\odot}$'")
    #plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
    plt.show(block=False)
