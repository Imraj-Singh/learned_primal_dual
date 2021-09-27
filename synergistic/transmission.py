# Copyright 2014-2020 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Phantoms typically used in transmission tomography."""

from __future__ import absolute_import, division, print_function
from os import path

import numpy as np

from odl.discr import DiscretizedSpace
from odl.phantom.geometric import ellipsoid_phantom
from odl import uniform_discr
import copy
import matplotlib.pyplot as plt

__all__ = ('shepp_logan_ellipsoids', 'shepp_logan', 'forbild')

def shepp_logan_channels(space, modified=False, min_pt=None, max_pt=None, n_channels = 1):
    rad18 = np.deg2rad(18.0)
    #       value  axisx  axisy     x       y  rotation
    ellipsoids= [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
            [-.98, .6624, .8740, 0.0000, -.0184, 0],
            [-.02, .1100, .3100, 0.2200, 0.0000, -rad18],
            [-.02, .1600, .4100, -.2200, 0.0000, rad18],
            [0.01, .2100, .2500, 0.0000, 0.3500, 0],
            [0.01, .0460, .0460, 0.0000, 0.1000, 0],
            [0.01, .0460, .0460, 0.0000, -.1000, 0],
            [0.01, .0460, .0230, -.0800, -.6050, 0],
            [0.01, .0230, .0230, 0.0000, -.6060, 0],
            [0.01, .0230, .0460, 0.0600, -.6050, 0]]
    n = n_channels
    z = 1
    channels = []
    for c in range(n_channels):
        for i in range(n):
            z += 1
            np.random.seed(z)
            ellipsoids[i][0] = (np.random.rand() - 0.5) * np.random.exponential(0.4)
        channels.append(copy.deepcopy(ellipsoids))
        print(np.shape(channels))
    if n_channels == 1:
        x = ellipsoid_phantom(space, channels)
    else:
        x = []
        for c in range(n_channels):
                x.append(ellipsoid_phantom(space, channels[c]))

    return x

size = 128

space = uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')
n_channels = 4
phantom = shepp_logan_channels(space, True, n_channels=n_channels)

print(np.shape(phantom))
fig=plt.figure()
axes = []
for i in range(n_channels):
    axes.append( fig.add_subplot(1, n_channels, i+1) )
    subplot_title=("Subplot"+str(i))
    axes[-1].set_title(subplot_title)  
    plt.imshow(phantom[i])
fig.tight_layout()
plt.show()
