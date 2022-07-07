#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:27:04 2019

@authors: Luis Alberto Canizares (adapted from benmosley)


"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

import pymc3 as pm
from scipy.stats import gaussian_kde
import arviz as az
from astropy.constants import c, m_e, R_sun, e, eps0, au
from math import sqrt, radians

try:
    import heliospacecraftlocation as hsl
except:
    print("Warning heliospacecraftlocation failed to connect with NASA")
    pass

class BayesianTOAPositioner:
    """Class for carrying out Bayesian TOA positioning.
    Requires at least 4 stations (to solve for x,y,v,t1).
    """
    
    def __init__(self,
                 stations,
                 x_lim=1.5*au.value,# maximum box size (m)
                 v_mu=c.value,# mean of velocity prior (m/s)
                 v_sd=(10/100)*c.value,# standard deviation of velocity prior (m/s)
                 t_cadence=60): # Spacecraft Cadence in seconds

        t_sd = t_cadence / 2  # standard deviation of observed values (s)
        t_lim =  24*60*60#np.sqrt(2)*x_lim/v_mu# resulting max tdoa value, used for t1 limit (s)
        
        # check if well posed
        if len(stations)<4:
            print("WARNING: at least 4 stations recommended for bayesian tdoa positioning!")

        
        self.x_lim = x_lim
        self.v_mu = v_mu
        self.v_sd = v_sd
        self.t_cadence = t_cadence
        self.t_sd = t_sd
        self.t_lim = t_lim
        self.stations = stations
        
    def sample(self, tdoa, draws=2000, tune=2000, chains=4,cores=4, init='jitter+adapt_diag', progressbar=True, verbose=True):
        "Carry out Bayesian inference"
        
        x_lim = self.x_lim
        v_mu = self.v_mu
        v_sd = self.v_sd
        t_sd = self.t_sd
        t_lim = self.t_lim
        stations = self.stations
        
        # assert correct number of observations
        if len(tdoa) != len(stations):
            raise Exception("ERROR: number of observations must match number of stations! (%i, %i)"%(len(tdoa), len(stations)))
        
        # assert max tdoa is not larger than t_lim
        if np.max(tdoa) > t_lim: 
            raise Exception("ERROR: tdoa > t_lim")

        with pm.Model():
        
            # Priors
            v = pm.TruncatedNormal("v", mu=v_mu, sigma=v_sd, upper=v_mu)
            x = pm.Uniform("x", lower=-x_lim, upper=x_lim, shape=2)          # prior on the source location (m)
            t0 = pm.Uniform("t0", lower=-t_lim, upper=t_lim)             # prior on the time offset (s)

            # Physics
            d = pm.math.sqrt(pm.math.sum((stations - x)**2, axis=1))         # distance between source and receivers
            t1 = d/v                                                         # time of arrival (TOA) of each receiver
            t = t1-t0                                                        # time of arrival (TOA) from the time offset
            
            # Observations
            #Y_obs = pm.Normal('Y_obs', mu=t, sd=t_sd, observed=tdoa)         # we assume Gaussian noise on the TDOA measurements
            pm.Normal('Y_obs', mu=t, sd=t_sd, observed=tdoa)
            # Posterior sampling
            #step = pm.HamiltonianMC()
            trace = pm.sample(draws=draws, tune=tune, chains=chains,cores = cores, target_accept=0.95, init=init, progressbar=progressbar,return_inferencedata=False)#, step=step)# i.e. tune for 1000 samples, then draw 5000 samples
            
            summary = az.summary(trace)
        
        mu = np.array(summary["mean"])
        sd = np.array(summary["sd"])
        
        if verbose:
            print("Percent divergent traces: %.2f %%"%(trace['diverging'].nonzero()[0].size / len(trace) * 100))
        
        return trace, summary, mu, sd
    
    def fit_xy_posterior(self, trace):
        """Helper function to estimate mu and sd of samples from a distribution,
        designed for when the tails of the distributions are large or non-zero"""
        
        # take mu to be the maximum of the posterior
        # take sigma to be the kde's half-width at 0.6065 (=normal distribution value at x=sigma)
        r = np.linspace(-self.x_lim, self.x_lim, 1000)
        ds = [gaussian_kde(trace['x'][:,i])(r) for i in range(2)]
        mu = [r[np.argmax(d)] for d in ds]
        widths = [r[d>0.6065*np.max(d)] for d in ds]
        sd = [(np.max(width)-np.min(width))/2. for width in widths]
        
        return mu, sd
    
    
    def forward(self, x, v=c.value):
        "predict time of flight for given source position"

        d = np.linalg.norm(self.stations-x, axis=1)
        t1 = d/v# time of flight values
        return t1




def triangulate(coords, times,t_cadence=60, cores=4, progressbar=True, report=0, plot=0):
    """"
    Input: spacecraft : array (Nx2) N is number of stations
        First column is time, Second column is location.
    """
    B = BayesianTOAPositioner(coords, t_cadence=t_cadence)
    trace, summary, _, _ = B.sample(times, draws=2000, tune=2000, chains=4, cores=cores, progressbar=progressbar)



    # analysis
    mu, sd = B.fit_xy_posterior(trace)
    t1_pred = B.forward(mu)


    if report == 1:
        # report
        print(summary)



    if plot == 1:
        # trace plot
        left = 0.125  # the left side of the subplots of the figure
        right = 0.9  # the right side of the subplots of the figure
        bottom = 0.1  # the bottom of the subplots of the figure
        top = 0.9  # the top of the subplots of the figure
        wspace = 0.2  # the amount of width reserved for blank space between subplots
        hspace = 0.5  # the amount of height reserved for white space between subplots

        ax = az.plot_trace(trace, compact=False)
        plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

        ax[1, 0].hlines(0.6065 * ax[1, 0].get_ylim()[1], mu[0] - sd[0],
                        mu[0] + sd[0])  # add mu, sigma lines to x,y plots
        ax[2, 0].hlines(0.6065 * ax[2, 0].get_ylim()[1], mu[1] - sd[1], mu[1] + sd[1])
        ax[1, 0].title.set_text('x0')
        ax[2, 0].title.set_text('x1')
        ax[1, 1].title.set_text('x0')
        ax[2, 1].title.set_text('x1')
        # plt.savefig("bayes_positioner_result1.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
        # pm.autocorrplot(trace)
        # pm.plot_posterior(trace)

        # local map
        plt.figure(figsize=(5, 5))
        spacecraft = coords / R_sun.value

        plt.scatter(spacecraft[:, 0], spacecraft[:, 1], marker="^", s=80, label="Spacecraft")
        ell = matplotlib.patches.Ellipse(xy=(mu[0] / R_sun.value, mu[1] / R_sun.value),
                                         width=4 * sd[0] / R_sun.value, height=4 * sd[1] / R_sun.value,
                                         angle=0., color='black', lw=1.5)
        plt.plot(mu[0] / R_sun.value, mu[1] / R_sun.value, 'k*')
        ell.set_facecolor('none')
        plt.gca().add_patch(ell)
        # plt.legend(loc=1)
        # plt.xlim(-1050, 1050)
        # plt.ylim(-1050, 1050)
        plt.xlabel("'HEE - X / $R_{\odot}$'")
        plt.ylabel("'HEE - Y / $R_{\odot}$'")
        # plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
        plt.show(block=False)

    return mu, sd, t1_pred, trace, summary

def animation_frame(xy, t_cadence):

    x, y = xy
    x_true = np.array([x * R_sun.value, y * R_sun.value])  # true source position (m)
    v_true = c.value  # speed of light (m/s)
    t0_true = 100  # source time. can be any constant, as long as it is within the uniform distribution prior on t0
    d_true = np.linalg.norm(stations - x_true, axis=1)
    t1_true = d_true / v_true  # true time of flight values
    # t_obs = t1_true-t0_true# true time difference of arrival values
    t_obs = t1_true
    np.random.seed(1)
    # t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations

    # sample
    np.random.seed(1)
    B = BayesianTOAPositioner(stations, t_cadence=t_cadence)
    trace, summary, _, _ = B.sample(t_obs, draws=2000, tune=2000, chains=4)

    # analysis
    mu, sd = B.fit_xy_posterior(trace)
    t1_pred = B.forward(mu)

    ell = matplotlib.patches.Ellipse(xy=(mu[0]/R_sun.value, mu[1]/R_sun.value),
              width=4*sd[0]/R_sun.value, height=4*sd[1]/R_sun.value,
              angle=0., color='black', label="Posterior ($2\sigma$)", lw=1.5)
    # detected_source = axs.plot(mu[0]/R_sun.value, mu[1]/R_sun.value,'k*')
    detected_source.set_xdata(mu[0]/R_sun.value)
    detected_source.set_ydata(mu[1]/R_sun.value)



    # true_source = axs.scatter(x_true[0]/R_sun.value, x_true[1]/R_sun.value,s=40,c='#e6b925', label="True source position")
    true_source.set_xdata(x_true[0]/R_sun.value)
    true_source.set_ydata(x_true[1]/R_sun.value)

    ellipse.set_center(xy=(mu[0] / R_sun.value, mu[1] / R_sun.value))
    ellipse.width = 4 * sd[0] / R_sun.value
    ellipse.height = 4 * sd[1] / R_sun.value
    ellipse.angle = 0.


    return detected_source, ell, true_source,


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    print('Running on PyMC3 v{}'.format(pm.__version__))
    save_vid = False
    
    # generate some test data
    N_STATIONS = 4
    np.random.seed(1)
    stations = np.random.randint(-200,200, size=(N_STATIONS,2))# station positions (m)



    # TEST STATIONs
    #stations = np.array([[-200, 200], [200, 200], [200, -200], [-200,-200]])
    #stations = np.array([[-200, 150], [146, 200], [125, -243], [-200,-200]])
    #stations = np.array([[700, 0], [546, 600], [525, -743], [600, 600]])
    # stations = np.array([[-26.00710538, -165.30659791], [110.77646426, -174.16680856], [215, 0]])   # 07/10/2020
    #stations = np.array([[-26.00710538, -165.30659791], [110.77646426, -174.16680856], [215, 0]])   # 07/10/2020
    #stations = np.array([[-26.00710538, -165.30659791], [-110.77646426, 174.16680856], [215, 0]])
    #stations = np.array([[-26.00710538, -165.30659791], [110.77646426, -174.16680856], [215, 0],[-96.43560481,152.72174817]])   # 07/10/2020 w/ solo


    # stations = np.array([[200, -200], [-200, -200], [0, 300]])
    #
    #
    # solarsystem = hsl.HelioSpacecraftLocation(date=[2020,7,11], objects=["psp","stereo_a","wind", "solo"])
    # stations = np.array(solarsystem.locate())




    # SURROUND
    #############################################################################
    L1 = [0.99 * (au / R_sun), 0]
    L4 = [(au / R_sun) * np.cos(radians(60)), (au / R_sun) * np.sin(radians(60))]
    L5 = [(au / R_sun) * np.cos(radians(60)), -(au / R_sun) * np.sin(radians(60))]
    ahead = [0, (au / R_sun)]
    behind = [0, -(au / R_sun)]

    stations_rsun = np.array([L1, L4, L5, ahead, behind])
    stations = stations_rsun
    N_STATIONS = len(stations)
    #############################################################################
    stations = stations*R_sun.value


    x_true = np.array([1*R_sun.value,1.2*R_sun.value])# true source position (m)
    v_true = c.value# speed of light (m/s)
    t0_true = 100 # source time. can be any constant, as long as it is within the uniform distribution prior on t0
    d_true = np.linalg.norm(stations-x_true, axis=1)
    t1_true = d_true/v_true# true time of flight values
    # t_obs = t1_true-t0_true# true time difference of arrival values
    t_obs = t1_true
    np.random.seed(1)
    t_obs = t_obs+0.05*np.random.randn(*t_obs.shape)# noisy observations


    mu, sd, t1_pred, trace, summary = triangulate(stations, t_obs,report=1,plot=1, t_cadence = 60)

    # report
    print(summary)
    print(f"t0_true: {t0_true}")
    print(f"t_obs:    {t_obs}")
    print(f"t1_true: {t1_true}")
    print(f"t1_pred: {t1_pred}")
    print(f"stations: {stations}")









    ## Take 2

    # local map

    # figure, axs = plt.subplots(figsize=(10,10))
    # axs.set_aspect('equal')
    # axs.scatter(stations[:,0], stations[:,1], marker="^", s=80, label="Receivers")
    # true_source, = axs.plot(x_true[0], x_true[1], "b*",label="True source position")
    # ell = matplotlib.patches.Ellipse(xy=(mu[0]/R_sun.value, mu[1]/R_sun.value),
    #           width=4*sd[0]/R_sun.value, height=4*sd[1]/R_sun.value,
    #           angle=0., color='black', label="Posterior ($2\sigma$)", lw=1.5)
    # detected_source, = axs.plot(mu[0]/R_sun.value, mu[1]/R_sun.value,'k*')
    # ell.set_facecolor('none')
    # ellipse = axs.add_patch(ell)
    # #plt.legend(loc=1)
    # #plt.xlim(-1050, 1050)
    # #plt.ylim(-1050, 1050)
    # axs.set_xlabel("'HEE - X / $R_{\odot}$'")
    # axs.set_ylabel("'HEE - Y / $R_{\odot}$'")
    # plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
    # # plt.show(block=False)
    #
    #
    # stations = stations*R_sun.value
    # x_true = x_true*R_sun.value
    #
    #
    # no_of_frames = 100
    # frames_burst = np.column_stack((np.linspace(0, -300, no_of_frames), np.linspace(0, 500, no_of_frames)))
    # animation = FuncAnimation(figure, func=animation_frame, frames=(frames_burst,t_cadence), interval=10)
    #
    # Writer = writers["ffmpeg"]
    # writer = Writer(fps=15, metadata={'artist': 'Me'}, bitrate=1800)
    #
    #
    # if save_vid==True:
    #     animation.save('animation_burst3.mp4', writer)
