import ROOT
import numpy as np
import root_numpy as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ROOT import TVector3

from python.utils import transform_into_xy, read_root_measurement, board_alignment, fit_trace

# define some global variables values in µm
x_chip = 8100
y_chip = 8100
# chip
delta_x_chip = np.full(52, 150)
delta_x_chip[0] = 300
delta_x_chip[51] = 300

delta_y_chip = np.full(160, 100)
delta_y_chip[79] = 200
delta_y_chip[80] = 200

delta_z = -0.01 * 10 ** 6


def fit_2(hit_1, hit_2):
    s = hit_1
    t = hit_2 - hit_1

    v = []
    for x in range(-5, 5):
        u = TVector3(t.X() * x, t.Y() * x, t.Z() * x)
        v.append(s + u)

    return v


def vec_between_hits(hit_1, hit_2):
    return (hit_2 - hit_1).Theta()


vectors = []
angles = []
angles_aligned = []
hit_trace = []
norm = []
norm_trace = []

board_distance = 10000 # 1,2 cm

for i in range(0, 295):
    print("Load File: {}".format(i))
    events_up, events_down = read_root_measurement('m4587', 'm4520', i)
    hits_up = []
    hits_down = []
    fits = []

    for i in range(len(events_up)):
        # for event in events_up:
        hit_up = transform_into_xy(events_up[i], board=1)
        hit_down = transform_into_xy(events_down[i], board=2)
        hits_up.append(hit_up)
        hits_down.append(hit_down)

        norm_vec = TVector3(hit_down.X(), hit_down.Y(), board_distance)
        muon_vec = TVector3(hit_down.X()-hit_up.X(), hit_down.Y()-hit_up.Y(), board_distance)
        x = hit_down.X()-hit_up.X()
        y = hit_down.Y()-hit_up.Y()
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan(r/board_distance)


        print("x: {}, y: {}".format(hit_down.X()-hit_up.X(), hit_down.Y()-hit_up.Y()))


        #angles.append(norm_vec.Angle(muon_vec))
        angles.append(theta)
        angles_aligned.append(np.pi/2-(vec_between_hits(hit_up, hit_down)))
        hit_trace.append(fit_trace(hit_up, hit_down))
        norm_trace.append(fit_trace(hit_down, hit_down))

        hit_down = board_alignment(hit_down)
        norm_vec = TVector3(hit_down.X(), hit_down.Y(), board_distance)
        muon_vec = TVector3(hit_down.X() - hit_up.X(), hit_down.Y() - hit_up.Y(), board_distance)

        angles_aligned.append(norm_vec.Angle(muon_vec))




        #fits.append(fit_2(hit_up, hit_down))

        #vectors.append(vec_between_hits(hit_up, hit_down))

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot([x.X() for x in hits_up],[x.Y() for x in hits_up],[x.Z() for x in hits_up],'bs')
    # ax.plot([x.X() for x in hits_down],[x.Y() for x in hits_down],[x.Z() for x in hits_down],'r*')
    # for fit in fits:
    #    ax.plot([x.X() for x in fit], [x.Y() for x in fit], [x.Z() for x in fit], 'black')
#
# x_surf = np.arange(-0, 8*8100, 1000)
# y_surf = np.arange(-0, 2*8100, 1000)
# x_surf, y_surf = np.meshgrid(x_surf, y_surf)
# z_surf = np.zeros(x_surf.shape)
# ax.plot_surface(x_surf, y_surf, z_surf, color='b',alpha=0.4)
# ax.plot_surface(x_surf, y_surf, z_surf+delta_z, color='r', alpha=0.4)
#
#
# plt.ylim(ymax=-100, ymin=16300)
# plt.xlim(xmax=-100, xmin=8*8100+100)
# ax.set_zlim(delta_z-100,100)
# ax.grid(False)
# plt.show()
print(angles)
#print(len(vectors))
print(len(angles))
fig = plt.figure()
#plt.hist(vectors, bins=100)
plt.hist(angles, bins=100, alpha=0.5)
plt.hist(angles_aligned, bins=100, alpha=0.5, label="aligned")


#ax = fig.add_subplot(111, projection='3d')
#ax.plot([x.X() for x in hits_up],[x.Y() for x in hits_up],[x.Z() for x in hits_up],'bs')
#ax.plot([x.X() for x in hits_down],[x.Y() for x in hits_down],[x.Z() for x in hits_down],'r*')

#i=0
#for trace in hit_trace:
#    ax.text(hits_up[i].X(), hits_up[i].Y(), 0, s=str(angles[i]))
#    ax.plot([x.X() for x in trace], [x.Y() for x in trace], [x.Z() for x in trace], 'black')
#    ax.plot([x.X() for x in norm_trace[i]], [x.Y() for x in norm_trace[i]], [x.Z() for x in trace], 'black')
 #  # print(hits_up[i].Angle(norm_trace[i]))
 #   i+=1

#x_surf = np.arange(-0, 8*8100, 1000)
#y_surf = np.arange(-0, 2*8100, 1000)
#x_surf, y_surf = np.meshgrid(x_surf, y_surf)
#z_surf = np.zeros(x_surf.shape)
#ax.plot_surface(x_surf, y_surf, z_surf, color='b',alpha=0.4)
#ax.plot_surface(x_surf, y_surf, z_surf+delta_z, color='r', alpha=0.4)


#plt.ylim(ymax=-100, ymin=16300)
#plt.xlim(xmax=-100, xmin=8*8100+100)
#ax.set_zlim(delta_z-100,100)
#ax.grid(False)


plt.legend()


plt.show()
