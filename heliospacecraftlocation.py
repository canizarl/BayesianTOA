import heliopy.data.spice as spicedata
import heliopy.spice as spice
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta
import astropy.units as u
from astropy.constants import c, m_e, R_sun, e, eps0, au
import numpy as np
import sys

from sunpy.time import parse_time
from sunpy.coordinates import get_horizons_coord




class HelioSpacecraftLocation:
    # print(f"Warning: Due to deprecation of heliopy, HelioSpacecraftLocation might fail. "
    #       f"Recommend using hsl which uses sunpy.")
    def __init__(self, date=[], objects=[""], orbit=0,orbitlength=1):
        self.date = date
        self.objects = objects
        self.orbit = orbit
        self.orbitlength = orbitlength




    def locate(self):
        date = self.date
        objects = self.objects
        orbit = self.orbit
        orbitlength = self.orbitlength




        locations = []

        # Constants
        r_sun = R_sun.value  # km
        AU = au.value  # km

        day = date[2]
        month = date[1]
        year = date[0]

        starttime = dt(year, month, day)
        endtime = starttime + timedelta(days=orbitlength)
        times = []
        while starttime < endtime:
            times.append(starttime)
            starttime += timedelta(hours=6)

        if "sun" in objects:
            sun_x = 0
            sun_y = 0
            locations.append([sun_x,sun_y])

        if "earth" in objects:
            # Earth location
            earth_coord.cartesian * (AU / r_sun)
            Earth_HEE_x = AU / r_sun
            Earth_HEE_y = 0
            locations.append([Earth_HEE_x,Earth_HEE_y])

        if "venus" in objects:
            # VENUS POSITION
            kernels = spicedata.get_kernel('planet_trajectories')
            spice.furnish(kernels)
            venus = spice.Trajectory('Venus')
            venus.generate_positions(times, 'Sun', 'HEE')
            venus.change_units(u.solRad)
            locations.append([venus.x[0].value,venus.y[0].value])

        if "psp" in objects:
            # PSP location
            kernels = spicedata.get_kernel('psp')
            kernels += spicedata.get_kernel('psp_pred')
            spice.furnish(kernels)
            psp = spice.Trajectory('SPP')

            psp.generate_positions(times, 'Sun', 'HEE')
            psp.change_units(u.solRad)
            locations.append([psp.x[0].value,psp.y[0].value])

            ##

        if "solo" in objects:
            kernels = spicedata.get_kernel('solo')
            spice.furnish(kernels)
            solo = spice.Trajectory('Solar Orbiter')

            solo.generate_positions(times, 'Sun', 'HEE')
            solo.change_units(u.solRad)
            locations.append([solo.x[0].value,solo.y[0].value])

            ##


        if "stereo_a"in objects:
            # STEREO A POSITION
            kernels = spicedata.get_kernel('stereo_a')
            kernels += spicedata.get_kernel('stereo_a_pred')
            spice.furnish(kernels)
            stereo_a = spice.Trajectory('Stereo Ahead')
            stereo_a.generate_positions(times, 'Sun', 'HEE')
            stereo_a.change_units(u.solRad)
            locations.append([stereo_a.x[0].value,stereo_a.y[0].value])

            ##

        if "stereo_b" in objects:
            # STEREO B POSITION
            kernels = spicedata.get_kernel('stereo_b')
            kernels += spicedata.get_kernel('stereo_b_pred')
            spice.furnish(kernels)
            stereo_b = spice.Trajectory('Stereo Behind')
            stereo_b.generate_positions(times, 'Sun', 'HEE')
            stereo_b.change_units(u.solRad)
            locations.append([stereo_b.x[0].value,stereo_b.y[0].value])

            ##

        if "wind" in objects:
            # wind location is in Sun - Earth L1
            wind_HEE_x = 0.99 * AU / r_sun
            wind_HEE_y = 0
            locations.append([wind_HEE_x,wind_HEE_y])

        return locations


class hsl:
    def __init__(self, date=[], objects=[""], orbit=0,orbitlength=1, timeres=24):
        self.date = date
        self.objects = objects
        self.orbit = orbit
        self.orbitlength = orbitlength
        self.timeres = timeres        # in hours




    def locate(self):
        date = self.date
        objects = self.objects
        orbit = self.orbit
        orbitlength = self.orbitlength
        timeres = self.timeres


        locations = []

        # Constants
        r_sun = R_sun.value  # km
        AU = au.value  # km

        day = date[2]
        month = date[1]
        year = date[0]


        starttime = dt(year, month, day)
        endtime = starttime + timedelta(days=orbitlength)
        times = []
        while starttime < endtime:
            times.append(starttime)
            starttime += timedelta(hours=timeres)

        if "sun" in objects:
            sun_x = 0
            sun_y = 0
            locations.append([sun_x,sun_y])

        if "mercury" in objects:
            #Mercury location
            mercury_coord = get_horizons_coord("Mercury Barycenter", times)
            mercury_xyz = mercury_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([mercury_xyz[0][0], mercury_xyz[1][0]])

        if "earth" in objects:
            # Earth location
            earth_coord = get_horizons_coord("Earth-Moon Barycenter", times)
            earth_xyz = earth_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value*(AU/r_sun)
            locations.append([earth_xyz[0][0],earth_xyz[1][0]])

        if "venus" in objects:
            # VENUS POSITION
            venus_coord = get_horizons_coord("Venus Barycenter", times)
            venus_xyz = venus_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([venus_xyz_xyz[0][0],venus_xyz[1][0]])

        if "psp" in objects:
            # PSP location
            global psp_xyz
            psp_coord = get_horizons_coord("PSP", times)
            psp_xyz = psp_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([psp_xyz[0][0],psp_xyz[1][0]])
            ##

        if "solo" in objects:
            # 2020-FEB-10 04:56:58.8550
            solo_coord = get_horizons_coord("SOLO", times)
            solo_xyz = solo_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([solo_xyz[0][0],solo_xyz[1][0]])
            ##


        if "stereo_a"in objects:
            # STEREO A POSITION
            stereoa_coord = get_horizons_coord("STEREO-A", times)
            stereoa_xyz = stereoa_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([stereoa_xyz[0][0],stereoa_xyz[1][0]])

            ##

        if "stereo_b" in objects:
            # STEREO B POSITION
            stereob_coord = get_horizons_coord("STEREO-B", times)
            stereob_xyz = stereob_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([stereob_xyz[0][0],stereob_xyz[1][0]])
            ##

        if "wind" in objects:
            # wind location is in Sun - Earth L1
            wind_coord = get_horizons_coord("WIND", times)
            wind_xyz = wind_coord.heliocentricearthecliptic.cartesian.get_xyz()[:].value * (AU / r_sun)
            locations.append([wind_xyz[0][0],wind_xyz[1][0]])

        return locations


    def help(self):
        print(f"hsl returns an array with positions of objects in solar sytem\n"
              f"objects: sun mercury venus earth psp solo stereo_a stereo_b wind\n"
              f"example\n"
              f"solarsystem = hsl(date=[year, month, day], objects=['psp', 'stereo_a', 'wind', 'solo'], timeres=24)\n"
              f"stations_rsun = np.array(solarsystem.locate())\n"
              f"\n"
              f"date: year:int\n"
              f"      \tmonth:int\n"
              f"      \tday:int\n"
              f"objects: list of strings with object ids\n"
              f"timeres:int  -  time resolution in hours for positions.\n"
              f"         \t24: 1 position every 24 hours\n"
              f"         \t1: 1 position for every hour of the day")





if __name__ == '__main__':

    # #########################
    # SETINGS
    # #########################
    if len(sys.argv)>1:
        day = int(sys.argv[0])
        month = int(sys.argv[1])
        year = int(sys.argv[2])
    else:
        day = 5
        month = 5
        year = 2022




    # toggle 1 (show)  or 0 (hide) the following objects.
    # Planets
    earth = 1   # HEE coordinates, by default at x = 1AU y=0
    venus = 1   # Not supported yet
    mercury = 0 # Not supported yet
    sun = 1     # HEE coordinates, by default at x = 0 y=0


    # Spacecraft
    parker_solar_probe = 1
    solar_orbiter = 1
    stereo_A = 1
    stereo_B = 1   # note: Spacecraft Not operational
    wind = 1

    # plot orbits? 1=yes 0=no
    plot_orbit = 0
    spacecraft_orbit_length = 100   #days

    objects = []
    if parker_solar_probe == 1: objects.append("psp")
    if solar_orbiter == 1: objects.append("solo")
    if stereo_A == 1: objects.append("stereo_a")
    if stereo_B == 0: objects.append("stereo_b")
    if wind == 1: objects.append("wind")
    if earth == 0: objects.append("earth")
    if venus == 0: objects.append("venus")
    if mercury == 0: objects.append("mercury")







    locations=[]
    # Constants
    r_sun = R_sun.value  # km
    AU = au.value  # km


    solarsystem = hsl(date=[year, month, day], objects=["psp", "stereo_a", "wind", "solo", "sun", "earth"], timeres=24)
    stations_rsun = np.array(solarsystem.locate())
    #
    # solarsystem2 = HelioSpacecraftLocation(date=[year, month, day], objects=["psp", "stereo_a", "wind", "solo"])
    # stations_rsun2 = np.array(solarsystem2.locate())

    #
    #
    #
    # plt.figure()
    # plt.scatter(stations_rsun[:,0],stations_rsun[:,1])
    # plt.show()
    #
    #







    #
    #
    #
    # starttime = dt(year, month, day)
    # endtime = starttime + timedelta(days=spacecraft_orbit_length)
    # times = []
    # while starttime < endtime:
    #     times.append(starttime)
    #     starttime += timedelta(hours=6)
    #
    # if "sun" in objects:
    #     sun_x = 0
    #     sun_y = 0
    #     locations.append([sun_x, sun_y])
    #
    # if "earth" in objects:
    #     # Earth location
    #     Earth_HEE_x = AU / r_sun
    #     Earth_HEE_y = 0
    #     locations.append([Earth_HEE_x, Earth_HEE_y])
    #
    # if "venus" in objects:
    #     # VENUS POSITION
    #     kernels = spicedata.get_kernel('planet_trajectories')
    #     spice.furnish(kernels)
    #     venus = spice.Trajectory('Venus')
    #     venus.generate_positions(times, 'Sun', 'HEE')
    #     venus.change_units(u.solRad)
    #     locations.append([venus.x[0].value, venus.y[0].value])
    #
    # if "psp" in objects:
    #     # PSP location
    #     kernels = spicedata.get_kernel('psp')
    #     kernels += spicedata.get_kernel('psp_pred')
    #     spice.furnish(kernels)
    #     psp = spice.Trajectory('SPP')
    #
    #     psp.generate_positions(times, 'Sun', 'HEE')
    #     psp.change_units(u.solRad)
    #     locations.append([psp.x[0].value, psp.y[0].value])
    #
    #     ##
    #
    # if "solo" in objects:
    #     kernels = spicedata.get_kernel('solo')
    #     spice.furnish(kernels)
    #     solo = spice.Trajectory('Solar Orbiter')
    #
    #     solo.generate_positions(times, 'Sun', 'HEE')
    #     solo.change_units(u.solRad)
    #     locations.append([solo.x[0].value, solo.y[0].value])
    #
    #     ##
    #
    # if "stereo_a" in objects:
    #     # STEREO A POSITION
    #     kernels = spicedata.get_kernel('stereo_a')
    #     kernels += spicedata.get_kernel('stereo_a_pred')
    #     spice.furnish(kernels)
    #     stereo_a = spice.Trajectory('Stereo Ahead')
    #     stereo_a.generate_positions(times, 'Sun', 'HEE')
    #     stereo_a.change_units(u.solRad)
    #     locations.append([stereo_a.x[0].value, stereo_a.y[0].value])
    #
    #     ##
    #
    # if "stereo_b" in objects:
    #     # STEREO B POSITION
    #     kernels = spicedata.get_kernel('stereo_b')
    #     kernels += spicedata.get_kernel('stereo_b_pred')
    #     spice.furnish(kernels)
    #     stereo_b = spice.Trajectory('Stereo Behind')
    #     stereo_b.generate_positions(times, 'Sun', 'HEE')
    #     stereo_b.change_units(u.solRad)
    #     locations.append([stereo_b.x[0].value, stereo_b.y[0].value])
    #
    #     ##
    #
    # if "wind" in objects:
    #     # wind location is in Sun - Earth L1
    #     wind_HEE_x = 0.99 * AU / r_sun
    #     wind_HEE_y = 0
    #     locations.append([wind_HEE_x, wind_HEE_y])
    #
    #
    #
    # fig, ax = plt.subplots()
    # ax.set_aspect('equal')
    # lim_plot = AU/r_sun + 15
    # ax.set(xlim=(-lim_plot, lim_plot), ylim=(-lim_plot, lim_plot))
    #
    #
    # if sun == 1:
    #     # circle for the sun
    #     sun = plt.Circle((0, 0), 3, color='gold', fill=True)
    #     ax.add_artist(sun)
    #
    # if parker_solar_probe == 1:
    #     psplocation = plt.plot(psp.x[0], psp.y[0], 'ro')
    #     plt.text(np.array(psp.x[0]) + 1, np.array(psp.y[0]) + 1, 'PSP')
    #     if plot_orbit == 1:
    #         plt.plot(psp.x, psp.y, 'r-')
    #
    # if solar_orbiter == 1:
    #     sololocation = plt.plot(solo.x[0], solo.y[0], 'ro')
    #     plt.text(np.array(solo.x[0]) + 1, np.array(solo.y[0]) + 1, 'Solar Orbiter')
    #     if plot_orbit == 1:
    #         plt.plot(solo.x, solo.y, 'r-')
    #
    #
    # if stereo_A == 1:
    #     stereo_a_location = plt.plot(stereo_a.x[0], stereo_a.y[0], 'ko')
    #     plt.text(np.array(stereo_a.x[0]) + 1, np.array(stereo_a.y[0]) + 1, 'Stereo A')
    #     if plot_orbit ==1:
    #         plt.plot(stereo_a.x, stereo_a.y, 'k-')
    # if stereo_B == 1:
    #     stereo_b_location = plt.plot(stereo_b.x[0], stereo_b.y[0], 'ko')
    #     plt.text(np.array(stereo_b.x[0]) + 1, np.array(stereo_b.y[0]) + 1, 'Stereo B')
    #     if plot_orbit ==1:
    #         plt.plot(stereo_b.x, stereo_b.y, 'k-')
    #
    # if wind == 1:
    #     windlocation = plt.plot(wind_HEE_x,wind_HEE_y, 'co')
    #     plt.text(wind_HEE_x - 20, wind_HEE_y + 1, 'wind')
    #
    # if earth == 1:
    #     earthlocation = plt.plot(Earth_HEE_x, Earth_HEE_y, 'bo')
    #     plt.text(Earth_HEE_x + 1, Earth_HEE_y + 1, 'Earth')
    #
    # if venus == 1:
    #     venus_location = plt.plot(venus.x[0], venus.y[0], 'go')
    #     plt.text(np.array(venus.x[0]) + 1, np.array(venus.y[0]) + 1, 'Venus')
    #     if plot_orbit ==1:
    #         plt.plot(venus.x, venus.y, 'k-')
    #
    # lim_plot = AU/r_sun + 15
    # ax.set(xlim=(-lim_plot, lim_plot), ylim=(-lim_plot, lim_plot))
    #
    #
    # month_strings = {
    #     1:'Jan',
    #     2:'Feb',
    #     3:'Mar',
    #     4:'Apr',
    #     5:'May',
    #     6:'Jun',
    #     7:'Jul',
    #     8:'Aug',
    #     9:'Sep',
    #     10:'Oct',
    #     11:'Nov',
    #     12:'Dec'}
    #
    # ax.set_title(f'Spacecraft Coordinates - {day} / {month_strings[month]} / {year}',fontsize=18)
    # ax.set_xlabel('HEE - X / $R_{\odot}$', fontsize=14)
    # ax.set_ylabel('HEE - Y / $R_{\odot}$', fontsize=14)
    # ax.grid()
    #
    # plt.show()
    #
    #





















