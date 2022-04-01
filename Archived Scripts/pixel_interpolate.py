import numpy as np
import matplotlib.pyplot as plt
from math import radians, sqrt

import pickle

from astropy.constants import c, m_e, R_sun, e, eps0, au
from scipy import interpolate

from mpl_toolkits.axes_grid1 import make_axes_locatable



if __name__ == "__main__":

    filenamef = "results_-300_340_Res_40_5stations.pkl"
    with open(filenamef, 'rb') as inp:
        results = pickle.load(inp)



    L1 = [0.99*(au/R_sun),0]
    L4 = [(au/R_sun)*np.cos(radians(60)),(au/R_sun)*np.sin(radians(60))]
    L5 = [(au/R_sun)*np.cos(radians(60)),-(au/R_sun)*np.sin(radians(60))]
    ahead = [0, (au/R_sun)]
    behind = [0, -(au/R_sun)]

    stations = np.array([L1, L4, L5, ahead, behind ])

    #stations = stations*R_sun.value


    xmapaxis = results[0]
    ymapaxis = results[1]
    delta_obs = results[2]
    scale_factor = 10




    # # Retest results for divergences
    # delta_obs[11,7] = 3.77394164   # first iteration failed tested again
    #
    # ii,jj  = np.where(xmapaxis == -100), np.where(ymapaxis==-20)
    # delta_obs[ii, jj] = 6.302300038911933
    #
    # ii,jj  = np.where(xmapaxis == 180), np.where(ymapaxis==60)
    # delta_obs[ii, jj] = 10.96671972375655
    #
    # ii,jj  = np.where(xmapaxis == 140), np.where(ymapaxis==-100)
    # delta_obs[ii, jj] = 1.7887262770034873
    #
    # ii, jj = np.where(xmapaxis == 180), np.where(ymapaxis == 100)
    # delta_obs[ii, jj] = 5.680299668882023
    #



    f = interpolate.interp2d(xmapaxis, ymapaxis, delta_obs, kind='linear')


    xnew = np.linspace(xmapaxis[0], xmapaxis[-1], xmapaxis.shape[0] * scale_factor)
    ynew = np.linspace(ymapaxis[0], ymapaxis[-1], ymapaxis.shape[0] * scale_factor)
    znew = f(xnew, ynew)


    """Remove artifacts"""

    znew[np.where(xnew>250),:] = 100


    # plt.figure(figsize=(8,8))
    # #plt.scatter(stations[:,0], stations[:,1], marker="^", s=80, label="Receivers")
    # #plt.scatter(x_true[0], x_true[1], s=40, label="True source position")
    # plt.pcolormesh(xmapaxis, ymapaxis, delta_obs.T, cmap='jet', vmin=0, vmax=30)
    # plt.colorbar()
    # plt.scatter(stations[:,0], stations[:,1],color = "k", marker="^", s=80, label="Spacecraft")
    # earth_orbit = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    # plt.gca().add_patch(earth_orbit)
    # plt.xlim(-300, 300)
    # plt.ylim(-300, 300)
    # plt.plot(au/R_sun,0,'bo', label="Earth")
    # plt.plot(0,0, 'yo', label="Sun")
    # plt.legend(loc=1)
    # plt.xlabel("'HEE - X / $R_{\odot}$'")
    # plt.ylabel("'HEE - Y / $R_{\odot}$'")
    # #plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
    # plt.axis("equal")
    # plt.show(block=False)

    # RESAMPLED

    # plt.figure(figsize=(8,8))
    # #plt.scatter(stations[:,0], stations[:,1], marker="^", s=80, label="Receivers")
    # #plt.scatter(x_true[0], x_true[1], s=40, label="True source position")
    # plt.pcolormesh(xnew, ynew, znew.T, cmap='jet', vmin=0, vmax=30)
    # plt.colorbar()
    # plt.scatter(stations[:,0], stations[:,1],color = "k", marker="^", s=80, label="Spacecraft")
    # earth_orbit = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    # plt.gca().add_patch(earth_orbit)
    # plt.xlim(-300, 300)
    # plt.ylim(-300, 300)
    # plt.plot(au/R_sun,0,'bo', label="Earth")
    # plt.plot(0,0, 'yo', label="Sun")
    # plt.legend(loc=1)
    # plt.xlabel("'HEE - X / $R_{\odot}$'")
    # plt.ylabel("'HEE - Y / $R_{\odot}$'")
    # #plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
    # plt.axis("equal")
    # plt.show(block=False)



    #####################################
    filenamef = "results_-300_340_Res_40_3stations.pkl"
    with open(filenamef, 'rb') as inp:
        results_3 = pickle.load(inp)

    stations_3 = np.array([L1, L4, L5])


    xmapaxis_3 = results_3[0]
    ymapaxis_3 = results_3[1]
    delta_obs_3 = results_3[2]


    f_3 = interpolate.interp2d(xmapaxis_3, ymapaxis_3, delta_obs_3, kind='linear')

    xnew_3 = np.linspace(xmapaxis_3[0], xmapaxis_3[-1], xmapaxis_3.shape[0] * scale_factor)
    ynew_3 = np.linspace(ymapaxis_3[0], ymapaxis_3[-1], ymapaxis_3.shape[0] * scale_factor)

    znew_3 = f_3(xnew_3, ynew_3)

    """Remove artifacts"""

    znew_3[np.where(xnew_3>240),:] = 100

    #
    # plt.figure(figsize=(8,8))
    # #plt.scatter(stations[:,0], stations[:,1], marker="^", s=80, label="Receivers")
    # #plt.scatter(x_true[0], x_true[1], s=40, label="True source position")
    # plt.pcolormesh(xmapaxis_3, ymapaxis_3, delta_obs_3.T, cmap='jet', vmin=0, vmax=30)
    # plt.colorbar()
    # plt.scatter(stations[:,0][0:3], stations[:,1][0:3],color = "k", marker="^", s=80, label="Spacecraft")
    # earth_orbit = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    # plt.gca().add_patch(earth_orbit)
    # plt.xlim(-300, 300)
    # plt.ylim(-300, 300)
    # plt.plot(au/R_sun,0,'bo', label="Earth")
    # plt.plot(0,0, 'yo', label="Sun")
    # plt.legend(loc=1)
    # plt.xlabel("'HEE - X / $R_{\odot}$'")
    # plt.ylabel("'HEE - Y / $R_{\odot}$'")
    # #plt.savefig("bayes_positioner_result2.jpg", bbox_inches='tight', pad_inches=0.01, dpi=300)
    # plt.axis("equal")
    # plt.show(block=False)
    #


    ##########################################################

    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    plt.subplots_adjust(top=1, bottom=0)
    im_0 = ax[0].pcolormesh(xnew_3, ynew_3, znew_3.T, cmap='jet', vmin=0, vmax=30)
    im_1 = ax[1].pcolormesh(xnew, ynew, znew.T, cmap='jet', vmin=0, vmax=30)

    earth_orbit = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    ax[0].add_patch(earth_orbit)
    earth_orbit_3 = plt.Circle((0, 0), au/R_sun, color='k', linestyle="dashed", fill=None)
    ax[1].add_patch(earth_orbit_3)

    ax[0].scatter(stations[:,0][0:3], stations[:,1][0:3],color = "w", marker="^",edgecolors="k", s=80, label="Spacecraft")
    ax[1].scatter(stations[:,0], stations[:,1],color = "w", marker="^",edgecolors="k", s=80, label="Spacecraft")

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')


    ax[0].set_xlim(-300, 300)
    ax[0].set_ylim(-300, 300)
    ax[1].set_xlim(-300, 300)
    ax[1].set_ylim(-300, 300)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.25, 0.01, 0.5])
    fig.colorbar(im_1, cax=cbar_ax)
    cbar_ax.set_ylabel('Triangulation uncertainty (Rsun)', fontsize=22)

    ax[0].plot(au/R_sun,0,'bo', label="Earth")
    ax[0].plot(0,0, 'yo', label="Sun")

    ax[1].plot(au/R_sun,0,'bo', label="Earth")
    ax[1].plot(0,0, 'yo', label="Sun")
    ax[0].legend(loc=1)
    ax[1].legend(loc=1)

    ax[0].set_title("3 Spacecraft configuration", fontsize=22)
    ax[1].set_title("5 Spacecraft configuration", fontsize=22)

    ax[0].set_xlabel("'HEE - X / $R_{\odot}$'", fontsize=22)
    ax[0].set_ylabel("'HEE - Y / $R_{\odot}$'", fontsize=22)
    ax[1].set_xlabel("'HEE - X / $R_{\odot}$'", fontsize=22)
    ax[1].set_ylabel("'HEE - Y / $R_{\odot}$'", fontsize=22)

    # Spacecraft Labels
    ax[0].text(0.72, 0.83, 'L4', horizontalalignment='center',
         verticalalignment='center', color="w", transform=ax[0].transAxes)
    ax[1].text(0.72, 0.83, 'L4', horizontalalignment='center',
         verticalalignment='center', color="w", transform=ax[1].transAxes)

    ax[0].text(0.72, 0.17, 'L5', horizontalalignment='center',
         verticalalignment='center', color="w", transform=ax[0].transAxes)
    ax[1].text(0.72, 0.17, 'L5', horizontalalignment='center',
         verticalalignment='center', color="w", transform=ax[1].transAxes)

    ax[0].text(0.9, 0.5, 'L1', horizontalalignment='center',
         verticalalignment='center', color="w", transform=ax[0].transAxes)
    ax[1].text(0.9, 0.5, 'L1', horizontalalignment='center',
         verticalalignment='center', color="w", transform=ax[1].transAxes)


    ax[1].text(0.5, 0.9, 'ahead', horizontalalignment='center',
         verticalalignment='center',color="w", transform=ax[1].transAxes)
    ax[1].text(0.5, 0.1, 'behind', horizontalalignment='center',
         verticalalignment='center',color="w", transform=ax[1].transAxes)

    ax[0].text(0.1, 0.9, 'a)', horizontalalignment='center',
               verticalalignment='center', color="w", transform=ax[0].transAxes, fontsize=22)
    ax[1].text(0.1, 0.9, 'b)', horizontalalignment='center',
               verticalalignment='center', color="w", transform=ax[1].transAxes, fontsize=22)

    plt.show(block=False)
