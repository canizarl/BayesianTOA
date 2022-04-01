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
import pickle

from math import sqrt, radians


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    print('Running on PyMC3 v{}'.format(pm.__version__))
    traceplotshow = 0

    # generate some test data
    #N_STATIONS = 3
    np.random.seed(1)


    L1 = [0.99*(au/R_sun),0]
    L4 = [(au/R_sun)*np.cos(radians(60)),(au/R_sun)*np.sin(radians(60))]
    L5 = [(au/R_sun)*np.cos(radians(60)),-(au/R_sun)*np.sin(radians(60))]
    ahead = [0, (au/R_sun)]
    behind = [0, -(au/R_sun)]

    stations = np.array([L1, L4, L5, ahead, behind])
    N_STATIONS = len(stations)

    stations = stations*R_sun.value

    delta_obs = []

    test_again = np.array([[-100, -20], [180,60], [140,-100],[180,100]])

    pixels = len(test_again)

    for each in test_again:
        x1 = each[0]
        y1 = each[1]
        if pixels == len(test_again):
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


            if traceplotshow == 1:
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
                #plt.savefig(f"traceplotpath/bayes_positioner_traceplot_{x1}_{y1}.jpg", bbox_inches='tight', pad_inches=0.01, dpi=100)
                #plt.close()
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



    results  = [test_again, delta_obs]
    with open(f'Manual_results_{test_again[0]}_{test_again[1]}_{N_STATIONS}stations.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)

    print("--------------RESULTS-----------")
    for each in results:
        print(f"P({each[0]}) =  {each[1]}    ")

