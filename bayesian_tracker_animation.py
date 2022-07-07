


from bayes_positioner import triangulate

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers

import pymc3 as pm
from scipy.stats import gaussian_kde
import arviz as az
from astropy.constants import c, m_e, R_sun, e, eps0, au
from math import sqrt, radians








if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    print('Running on PyMC3 v{}'.format(pm.__version__))




