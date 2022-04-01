from bayes_positioner import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

import os

import pymc3 as pm
from scipy.stats import gaussian_kde
from scipy.ndimage import median_filter


import arviz as az
from astropy.constants import c, m_e, R_sun, e, eps0, au
from heliospacecraftlocation import hsl
from heliospacecraftlocation import HelioSpacecraftLocation as HSL
import datetime as dt

from math import sqrt, radians
from joblib import Parallel, delayed
import multiprocessing

from contextlib import contextmanager
import sys, os
import logging

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def mkdirectory(directory):
    dir = directory
    isExist = os.path.exists(dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir)
        print("The new directory is created!")

def parallel_pos_map(x1,y1,stations,xrange,yrange,xres,yres, cores=1, traceplotsave=False, figdir=f"./traceplots", date_str="date"):
    tloop0 = dt.datetime.now()
    try:
        currentloop = f"""x, y : {x1},{y1}"""
        print(currentloop)

        x_true = np.array([x1 * R_sun.value, y1 * R_sun.value])  # true source position (m)
        v_true = c.value  # speed of light (m/s)
        t0_true = 100  # source time. can be any constant, as long as it is within the uniform distribution prior on t0
        d_true = np.linalg.norm(stations - x_true, axis=1)
        t1_true = d_true / v_true  # true time of flight values
        # t_obs = t1_true-t0_true# true time difference of arrival values
        t_obs = t1_true
        np.random.seed(1)
        # t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations

        # Make sure to use cores=1 when using parallel loops. cores in triangulate function refers to number of cores
        # used by the MCMC solver.
        mu, sd, t1_pred, trace, summary = triangulate(stations, t_obs, cores=1, progressbar=False, report=0, plot=0)

        # report
        # print(summary)
        # print(f"t0_true: {t0_true}")
        # print(f"t_obs:    {t_obs}")
        # print(f"t1_true: {t1_true}")
        # print(f"t1_pred: {t1_pred}")
        # print(f"stations: {stations}")

        if traceplotsave == True:
            traceplotpath = f"./traceplots/{date_str}/traceplot_{xrange[0]}_{xrange[1]}_{yrange[0]}_{yrange[1]}_{xres}_{yres}"
            mkdirectory(traceplotpath)

            # trace plot
            ax, ay = az.plot_trace(trace, compact=False)[1:3, 0]
            ax.hlines(0.6065 * ax.get_ylim()[1], mu[0] - sd[0], mu[0] + sd[0])  # add mu, sigma lines to x,y plots
            ay.hlines(0.6065 * ay.get_ylim()[1], mu[1] - sd[1], mu[1] + sd[1])

            figname = traceplotpath + f"/bayes_positioner_traceplot_{x1}_{y1}.jpg"
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.01, dpi=100)
            # plt.close()
            # pm.autocorrplot(trace)
            # pm.plot_posterior(trace)

        # difference detected observed
        x_true = x_true / R_sun.value
        xy = mu[0] / R_sun.value, mu[1] / R_sun.value

        delta_obs = sqrt((xy[0] - x_true[0]) ** 2 + (xy[1] - x_true[1]) ** 2)
        print(f"delta_obs: {delta_obs}")
        res = np.array([delta_obs, np.nan, np.nan])
        tloop1 = dt.datetime.now()

    except:
        delta_obs = 200
        print(f"SIM FAILED at P({x1},{y1}), SKIPPED")
        res = np.array([delta_obs, x1, y1])
        tloop1 = dt.datetime.now()
        pass

    tloopinseconds = (tloop1 - tloop0).total_seconds()
    print(f"Time Loop : {tloop1 - tloop0}   :   {tloopinseconds}s   ")

    return res

def plot_map_simple(delta_obs, xmapaxis, ymapaxis, stations,vmin=0,vmax=30, savefigure=False, showfigure=True, title="",figdir=f"./MapFigures", date_str="date"):
    xres = xmapaxis[1]-xmapaxis[0]
    yres = ymapaxis[1]-ymapaxis[0]
    N_STATIONS = len(stations)

    fig, ax = plt.subplots(1,1,figsize=(8,8))
    plt.subplots_adjust(top=1, bottom=0)

    im_0 = ax.pcolormesh(xmapaxis, ymapaxis, delta_obs.T, cmap='jet', vmin=vmin, vmax=vmax)

    earth_orbit = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    ax.add_patch(earth_orbit)

    ax.scatter(stations[:,0], stations[:,1],color = "w", marker="^",edgecolors="k", s=80, label="Spacecraft")

    ax.set_aspect('equal')

    ax.set_xlim(xmapaxis[0], xmapaxis[-1])
    ax.set_ylim(ymapaxis[0], ymapaxis[-1])

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.5])
    fig.colorbar(im_0, cax=cbar_ax)
    cbar_ax.set_ylabel('Triangulation uncertainty (Rsun)', fontsize=22)

    ax.plot(au / R_sun, 0, 'bo', label="Earth")
    ax.plot(0, 0, 'yo', label="Sun")

    ax.legend(loc=1)

    ax.set_xlabel("'HEE - X / $R_{\odot}$'", fontsize=22)
    ax.set_ylabel("'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax.set_title(title, fontsize=22)

    if savefigure == True:
        figdir = f"{figdir}/{date_str}"
        mkdirectory(figdir)
        plt.savefig(figdir+f"/bayes_positioner_map_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)

    if showfigure == True:
        plt.show(block=False)
    else:
        plt.close(fig)

def savedata(delta_obs, xmapaxis, ymapaxis, stations,dir=f"./Data", date_str="date"):
    import pickle
    mkdirectory(dir+f"/{date_str}")


    xres = xmapaxis[1]-xmapaxis[0]
    yres = ymapaxis[1]-ymapaxis[0]
    N_STATIONS = len(stations)

    results  = [xmapaxis, ymapaxis, delta_obs, stations]
    with open(dir+f"/{date_str}"+f'/results_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}stations.pkl', 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)

def loaddata(filenamef):
    import pickle

    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)

    xmapaxis = results[0]
    ymapaxis = results[1]
    delta_obs = results[2]
    stations = results[3]

    return xmapaxis, ymapaxis, delta_obs, stations

def interpolate_map(delta_obs, xmapaxis, ymapaxis, scale_factor=10, kind="linear"):
    from scipy import interpolate


    f = interpolate.interp2d(xmapaxis, ymapaxis, delta_obs.T, kind=kind)

    xnew = np.linspace(xmapaxis[0], xmapaxis[-1], xmapaxis.shape[0] * scale_factor)
    ynew = np.linspace(ymapaxis[0], ymapaxis[-1], ymapaxis.shape[0] * scale_factor)
    znew = f(xnew, ynew)

    return xnew,ynew, znew.T

def simulation_report(title="",xrange=[], yrange=[], xres=0, yres=0, pixels=0, coresused=0, tstart=dt.datetime.now(), tfinal=dt.datetime.now(),writelog=True):
    SIMREPORT = f""" 
    -------------------REPORT---------------------
    {title}
    Grid: X{xrange}, Y{yrange}
    xres: {xres}
    yres: {yres}
    totalpixels: {pixels}


    cores: {coresused}
    Computational cost: {tfinal - tstart}

    ----------------END-REPORT---------------------
    """
    logging.info(SIMREPORT)
    print(SIMREPORT)

def medfil(*args, **kwargs):
    new_image = median_filter(*args, **kwargs)
    return new_image



if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    string ='Running on PyMC3 v{}'.format(pm.__version__)
    print(string)

    ncores_cpu = multiprocessing.cpu_count()

    # generate some test data
    np.random.seed(1)
    day = 11
    month = 7
    year = 2020
    date_str = f"{year}_{month:02d}_{day:02d}"


    mkdirectory("./logs")
    logging.basicConfig(filename=f'logs/sim_{date_str}.log', level=logging.INFO)
    logging.info(string)
    logging.info(f'Date for Simulation: {date_str} ')


    if date_str == "surround":
        # SURROUND
        #############################################################################
        L1 = [0.99*(au/R_sun),0]
        L4 = [(au/R_sun)*np.cos(radians(60)),(au/R_sun)*np.sin(radians(60))]
        L5 = [(au/R_sun)*np.cos(radians(60)),-(au/R_sun)*np.sin(radians(60))]
        ahead = [0, (au/R_sun)]
        behind = [0, -(au/R_sun)]

        stations_rsun = np.array([L1, L4, L5, ahead, behind])
        #############################################################################
    elif date_str == "test":
        stations_rsun = np.array([[200, 200], [-200, -200], [-200, 200], [200, -200]])

    elif date_str == "manual":
        stations_rsun = np.array([[45.27337378, 9.90422281],[-24.42715218,-206.46280171],[ 212.88183411,0.]])
        date_str = f"{year}_{month:02d}_{day:02d}"
    else:
        solarsystem = hsl(date=[year, month, day], objects=["psp", "stereo_a", "wind", "solo"])
        stations_rsun = np.array(solarsystem.locate())






    N_STATIONS = len(stations_rsun)
    stations = stations_rsun*R_sun.value

    # Making grid
    xrange = [-200,400]
    xres = 5
    yrange = [-250, 250]
    yres = xres
    xmapaxis = np.arange(xrange[0], xrange[1], xres)
    ymapaxis = np.arange(yrange[0], yrange[1], yres)

    pixels0 = len(xmapaxis)*len(ymapaxis)
    print(f" Pixels : {pixels0}")

    tpl_l = 1.5   # time per loop in mins low end
    tpl_h = 5     # time per loop in mins high end
    est_low = pixels0*1.5/ncores_cpu
    est_high= pixels0*5/ncores_cpu

    print(f" Estimated Simulation time: {est_low} - {est_high} mins")

    """  ---------------------------------------------------------------------   """
    """  -------------------------  ON / OFF ---------------------------------   """
    runsimulation = False
    run_failed_again = False
    run_savedata = False
    run_loaddata = True
    run_plotdata = True
    run_datainterpolate = False  # Mantain False, no longer in use
    run_median_filter = True



    yesanswer = ["y".casefold(), "ye".casefold(), "yes".casefold()]
    noanswer = ["n".casefold(), "no".casefold()]


    filename = f"./Data/{date_str}/results_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}stations.pkl"
    #Doesnt make sense to save data or run flagged points if simulation is off.
    if runsimulation == False:
        run_savedata = False
        run_failed_again = False
        if run_loaddata == True:
            #Check if data exists
            isExist = os.path.exists(filename)
            if not isExist:
                # If data doesnt exist run simulation?
                runsimans = input("The data you are looking for does not exist. Run simulation? y/n:   ")
                if runsimans.casefold() in yesanswer:
                    runsimulation = True
                    runsimflagsans = input("Do you want to run failed points a second time? y/n:    ")
                    if runsimflagsans.casefold() in yesanswer:
                        run_failed_again = True
                    runsavedataans = input("Would you like to save the data? y/n:     ")
                    if runsavedataans.casefold() in yesanswer:
                        run_savedata = True
                run_loaddata = False

    if runsimulation == True:
        # Check if data exists
        isExist = os.path.exists(filename)
        if isExist:
            # If data exists run simulation?
            runsimans = input("There is data for this simulation. Are you sure you want to run? y/n:   ")
            if runsimans.casefold() in noanswer:
                runsimulation = False
                run_failed_again = False
                run_savedata = False
                run_loaddata = True
                run_median_filter = True


    """  ---------------------------------------------------------------------   """




    """  ---------------------------------------------------------------------   """
    """  ------------------------- Simulation --------------------------------   """
    """  ---------------------------------------------------------------------   """
    if runsimulation == True:
        compcost0 = dt.datetime.now()
        coresforloop = multiprocessing.cpu_count()


        results = Parallel(n_jobs=coresforloop, verbose=100)(delayed(parallel_pos_map)(i, j, stations=stations, xrange=xrange,
                                                                                       yrange=yrange,xres=xres,yres=yres,figdir=f"./traceplots",
                                                                                       date_str=date_str) for i in xmapaxis for j in ymapaxis)
        delta_obs=np.array(results)[:,0]
        flaggedpoints = np.array(results)[:,1:]
        compcost1 = dt.datetime.now()


        if (run_failed_again == True) and np.isfinite(flaggedpoints).any():
            print("RUNNING FAILED POINTS AGAIN")
            flagindex = []
            flagPx = []
            flagPy = []
            for i in range(0, len(flaggedpoints)):
                if ~np.isnan(flaggedpoints[i][0]) and ~np.isnan(flaggedpoints[i][1]):
                    flagindex.append(i)
                    flagPx.append([flaggedpoints[i][0]])
                    flagPy.append([flaggedpoints[i][1]])

            results_failed = Parallel(n_jobs=coresforloop, verbose=100)(delayed(parallel_pos_map)(flagPx[i], flagPy[i], stations=stations, xrange=xrange, yrange=yrange, xres=xres, yres=yres) for i in range(0,len(flagindex)))
            delta_obs[flagindex] = np.array(results_failed,dtype=object)[:,0]
            flaggedpoints = np.array(results_failed, dtype=object)[:, 1]







        simulation_report(title="Positioner Mapping Parallel", xrange=xrange, yrange=yrange, xres=xres, yres=yres, pixels=pixels0,
                          coresused=coresforloop, tstart=compcost0, tfinal=compcost1)

        delta_obs = np.reshape(delta_obs, (len(xmapaxis), len(ymapaxis)))

    """  ---------------------------------------------------------------------   """
    """  ------------------- End - of - Simulation ---------------------------   """
    """  ---------------------------------------------------------------------   """



    """  ------------------------ Save Data ----------------------------------   """
    if run_savedata == True:
        savedata(delta_obs, xmapaxis, ymapaxis, stations_rsun, dir=f"./Data", date_str=date_str)
    """  ---------------------------------------------------------------------   """


    """  ------------------------ Load Data ----------------------------------   """
    if run_loaddata == True:
        filename = f"./Data/{date_str}/results_{xmapaxis[0]}_{xmapaxis[-1]}_{ymapaxis[0]}_{ymapaxis[-1]}_{xres}_{yres}_{N_STATIONS}stations.pkl"
        xmapaxis, ymapaxis, delta_obs, stations_rsun = loaddata(filename)
    """  ---------------------------------------------------------------------   """



    """  ------------------------ PLOT Data ----------------------------------   """
    if run_plotdata == True:
        plot_map_simple(delta_obs, xmapaxis, ymapaxis, stations_rsun,vmax=100, savefigure=True, date_str=date_str)
    """  ---------------------------------------------------------------------   """

    """  ------------------------ Median Filter ---------------------------   """
    if run_median_filter == True:
    # from scipy.ndimage import median_filter as medfil
        median_filter_image = medfil(delta_obs, size=(6,6))
        plot_map_simple(median_filter_image, xmapaxis, ymapaxis, stations_rsun, vmax=100, savefigure=True, date_str=date_str)
    """  ---------------------------------------------------------------------   """



    # """  ------------------------ Interpolate Data ---------------------------   """
    # """ GENERATES ARTIFACTS. BETTER USE MEDIAN FILTER"""
    # if run_datainterpolate == True:
    #     xnew,ynew, znew = interpolate_map(new_image, xmapaxis, ymapaxis, scale_factor=2, kind="quintic")
    #     plot_map_simple(znew, xnew, ynew, stations_rsun,vmax=100, savefigure=True, date_str=date_str)
    #     savedata(znew,xnew,ymapaxis,stations_rsun,dir=f"./InterpData/", date_str=date_str)
    # """  ---------------------------------------------------------------------   """









