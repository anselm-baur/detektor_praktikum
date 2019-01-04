import ROOT
import numpy as np
import root_numpy as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ROOT import TVector3

from utils import *


angles = []
angles_aligned = []

num_measurements = [295,170]
for campaign in range(1,3):
    for i in range(num_measurements[campaign-1]):
        print("Load File: {}/{}".format(campaign, i))
        events_up, events_down = load_root_measurement(campaign, 'm4587', 'm4520', i)
        hits_up = []
        hits_down = []

        for i in range(len(events_up)):
            # for event in events_up:
            hit_up = transform_into_xy(events_up[i], board=1)
            hit_down = transform_into_xy(events_down[i], board=2)

            # for printing the traces
            hits_up.append(hit_up)
            hits_down.append(hit_down)

            x = hit_down.X() - hit_up.X()
            y = hit_down.Y() - hit_up.Y()
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan(r / delta_z)

            angles.append(theta)

            # calculate angle with board alignment
            hit_down = board_alignment(hit_down)
            x = hit_down.X() - hit_up.X()
            y = hit_down.Y() - hit_up.Y()
            r = np.sqrt(x ** 2 + y ** 2)
            theta = np.arctan(r / delta_z)

            angles_aligned.append(theta)


fig = plt.figure()

plt.hist(angles, bins=100, alpha=0.5, label="not aligned")
plt.hist(angles_aligned, bins=100, alpha=0.5, label="aligned")
plt.legend()
plt.show()