import os
import adler

#from phantom import random_phantom_channels

import tensorflow as tf
import numpy as np
import odl
import odl.contrib.tensorflow

from phantoms import shepp_logan_channels, random_phantom_channels

import matplotlib.pyplot as plt
# Create ODL data structures
size = 128
space = odl.uniform_discr([-64, -64], [64, 64], [size, size],
                          dtype='float32')
n_channels = 3

phantom = random_phantom_channels(space, n_channels=n_channels)

print(np.shape(phantom))
fig=plt.figure()
if n_channels == 1:
    plt.imshow(phantom)
if n_channels>1:
    axes = []
    for i in range(n_channels):
        axes.append( fig.add_subplot(1, n_channels, i+1) )
        subplot_title=("Subplot"+str(i))
        axes[-1].set_title(subplot_title)  
        plt.imshow(phantom[i])
    fig.tight_layout()
plt.show()
