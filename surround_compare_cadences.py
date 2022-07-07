import datetime

from bayes_positioner import triangulate

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from matplotlib.pyplot import cm

import pymc3 as pm
from scipy.stats import gaussian_kde
import arviz as az
from astropy.constants import c, m_e, R_sun, e, eps0, au
from math import sqrt, radians

import multiprocessing
import os

class typeIII:
  def __init__(self, freq, xy, sd):
    self.freq = freq
    self.xy = xy
    self.sd = sd

class triangulated_source:
    def __init__(self,t_cadence, mu, sd, stations=[], true_sources=[]):
        self.t_cadence = t_cadence
        self.mu = mu
        self.sd = sd
        self.stations=stations
        self.true_sources=true_sources


def savepickle(results, filename):
    import pickle
    with open(filename, 'wb') as outp:
        pickle.dump(results, outp, pickle.HIGHEST_PROTOCOL)


def loaddata(filenamef):
    import pickle

    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)
    return results
def mkdirectory(directory):
    dir = directory
    isExist = os.path.exists(dir)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(dir)
        print("The new directory is created!")


def cart2pol(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan(y/x)
    return(r,theta)
def pol2cart(r, phi):
    x = r* np.cos(phi)
    y = r* np.sin(phi)
    return(x, y)
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    print('Running on PyMC3 v{}'.format(pm.__version__))
    t_report_start = datetime.datetime.now()
    t_report_loops = []




    ###############################################################
    # Check if running in server or local
    machinenames = ['aodh', 'anode','bnode', 'cnode', 'dnode']
    if os.uname().nodename in machinenames:
        runserver = True
    else:
        runserver = False
    if runserver == True:
        run_sim = True
        run_save = True
        run_load = False
        run_plot = False
    else:
        run_sim = False
        run_save = False
        run_load = True
        run_plot = True
    ###############################################################




    ncores = multiprocessing.cpu_count()
    n_points = 10
    source_theta = np.radians(10)
    source_h_min = pol2cart(15,source_theta)
    source_h_max = pol2cart(150, source_theta)

    x_burst_rsun = np.linspace(source_h_min[0], source_h_max[0], n_points)
    y_burst_rsun = np.linspace(source_h_min[1],source_h_max[1],n_points)

    burst_xy = np.array(list(zip(x_burst_rsun*R_sun.value, y_burst_rsun*R_sun.value)))

    # x_burst_rsun = np.linspace(5, 150, n_points)
    # y_burst_rsun = np.linspace(0,40,n_points)
    #
    #
    # burst_xy = np.array(list(zip(x_burst_rsun*R_sun.value, y_burst_rsun*R_sun.value)))

    for theta_AB_deg in range(0, 360+1, 10):
        print(f"Theta AB: {theta_AB_deg}")
        t_loop_start = datetime.datetime.now()
        # SURROUND POSITIONS
        #############################################################################
        L1 = [0.99 * (au / R_sun), 0]
        L4 = [(au / R_sun) * np.cos(radians(60)), (au / R_sun) * np.sin(radians(60))]
        L5 = [(au / R_sun) * np.cos(radians(60)), -(au / R_sun) * np.sin(radians(60))]
        dh = 0.1
        # theta_AB_deg = 90
        theta_AB = np.radians(theta_AB_deg)
        ahead =  pol2cart((1-dh)*(au / R_sun),theta_AB)
        behind = pol2cart((1+dh)*(au / R_sun),-theta_AB)

        stations_rsun = np.array([L1, L4, L5, ahead])
        # stations_rsun = np.array([L1, L4, L5])
        stations = stations_rsun
        N_STATIONS = len(stations)
        stations = stations*R_sun.value
        #############################################################################


        v_true = c.value# speed of light (m/s)
        t0_true = 100  # source time. can be any constant, as long as it is within the uniform distribution prior on t0


        experiment = {}
        dir_out = f"SURROUND_CADENCES_EXPERIMENT_{N_STATIONS}_noBEHIND/2000Samples"
        mkdirectory(dir_out)
        fname =  f"{dir_out}/surround_cadence_experiment_{theta_AB_deg}deg.pkl"

        cadences = [10,20,30,40,50, 60]
        if run_sim == True:
            for cadence in cadences:
                mu_detected = []
                sd_detected = []
                t1_pred_detected = []
                trace_detected = []
                summary_detected = []
                for source in burst_xy:
                    x_true = source# true source position (m)
                    d_true = np.linalg.norm(stations-x_true, axis=1)
                    t1_true = d_true/v_true# true time of flight values
                    # t_obs = t1_true-t0_true# true time difference of arrival values
                    t_obs = t1_true
                    np.random.seed(1)
                    #t_obs = t_obs + 1*np.random.randn(*t_obs.shape)# noisy observations

                    mu, sd, t1_pred, trace, summary = triangulate(stations, t_obs, cores=ncores, report=0, plot=0, t_cadence=cadence)

                    mu_detected.append(mu)
                    sd_detected.append(sd)
                    t1_pred_detected.append(t1_pred)
                    trace_detected.append(trace)
                    summary_detected.append(summary)

                experiment[cadence] = triangulated_source(cadence, mu_detected, sd_detected, stations=stations_rsun, true_sources=x_true)
            if run_save == True:
                # Save results
                savepickle(experiment, fname)
            t_loop_end = datetime.datetime.now()
            t_report_loops.append([f'Angle: {theta_AB_deg}',str(t_loop_end-t_loop_start)])


        if run_load == True:
            # Load results
            experiment = loaddata(fname)







        if run_plot == True:
            # ################################################################## #
            #                         Plot
            # ################################################################## #
            r,theta = cart2pol(burst_xy[:,0]/R_sun.value,burst_xy[:,1]/R_sun.value)

            fig, ax = plt.subplot_mosaic("AB;AC")

            # set the spacing between subplots
            fig.subplots_adjust(left=0.1,
                                bottom=0.1,
                                right=0.9,
                                top=0.9,
                                wspace=0.4,
                                hspace=0.4)
            fig.set_size_inches(18.5, 7.5)
            ax['A'].set_aspect('equal')

            earth_orbit = plt.Circle((0, 0), au / R_sun, color='k', linestyle="dashed", fill=None)
            ax['A'].add_patch(earth_orbit)
            ax['A'].plot(au / R_sun, 0, 'bo', label="Earth")
            ax['A'].plot(0, 0, 'yo', label="Sun")

            ax['A'].scatter(stations_rsun[:, 0], stations_rsun[:, 1], color="w", marker="^", edgecolors="k", s=80,
                            label="Spacecraft")

            color = iter(cm.rainbow(np.linspace(0, 1, len(cadences))))
            for cadence in cadences:
                col = next(color)
                for i in range(0, len(experiment[cadence].mu)):
                    mu = experiment[cadence].mu[i]
                    sd = experiment[cadence].sd[i]

                    ell = matplotlib.patches.Ellipse(xy=(mu[0] / R_sun.value, mu[1] / R_sun.value),
                                                     width=4 * sd[0] / R_sun.value, height=4 * sd[1] / R_sun.value,
                                                     angle=0., color=col, lw=1, label=f"{cadence}s")
                    ax['A'].plot(mu[0] / R_sun.value, mu[1] / R_sun.value, 'k*')
                    ell.set_facecolor('none')
                    ax['A'].add_patch(ell)

            ax['A'].scatter(burst_xy[:, 0] / R_sun.value, burst_xy[:, 1] / R_sun.value, color="orange", marker="o",
                            edgecolors="k", s=80, label="True Sources")

            ax['A'].set_xlim(-250, 250)
            ax['A'].set_ylim(-250, 250)

            handles, labels = ax['A'].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax['A'].legend(by_label.values(), by_label.keys(), loc='upper left')
            ax['A'].set_xlabel("'HEE - X / $R_{\odot}$'", fontsize=22)
            ax['A'].set_ylabel("'HEE - Y / $R_{\odot}$'", fontsize=22)
            ax['A'].set_title("", fontsize=22)

            ax['B'].set_title(" Uncertainty X Direction")
            ax['B'].set_xlabel("R - [Rsun]")
            ax['B'].set_ylabel("2$\sigma$ Uncertainty [Rsun]")
            color = iter(cm.rainbow(np.linspace(0, 1, len(cadences))))
            for cadence in cadences:
                col = next(color)
                sd = experiment[cadence].sd
                sd = np.array(sd)
                ax['B'].plot(r, 4 * sd[:, 0] / R_sun.value, c=col, ls="-", label=f"{cadence}s")
            ax['B'].legend(loc='upper left')
            ax['B'].set_ylim(0, 50)

            ax['C'].set_title(" Uncertainty Y Direction")
            ax['C'].set_xlabel("R - [Rsun]")
            ax['C'].set_ylabel("2$\sigma$ Uncertainty [Rsun]")
            color = iter(cm.rainbow(np.linspace(0, 1, len(cadences))))
            for cadence in cadences:
                col = next(color)
                sd = experiment[cadence].sd
                sd = np.array(sd)
                ax['C'].plot(r, 4 * sd[:, 1] / R_sun.value, c=col, ls="-", label=f"{cadence}s")
            ax['C'].legend(loc='upper left')
            ax['C'].set_ylim(0, 50)

            figname = f"frame_{theta_AB_deg}.jpg"
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.01, dpi=100)
            plt.show(block=False)


        t_report_end = datetime.datetime.now()
        t_total = t_report_end-t_report_start

        if run_sim == True:
            report = f"""
            Finished: 
            Total time:{t_total}
            Time per loop:
            {t_report_loops}
            
            
            END
            """
            print(report)




