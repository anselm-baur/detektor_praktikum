import ROOT
import numpy as np
import root_numpy as rn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ROOT import TVector3

# define some global variables values in µm
x_chip = 8100
y_chip = 8100
# chip
delta_x_chip = np.full(52,150)
delta_x_chip[0] = 300
delta_x_chip[51] = 300

delta_y_chip = np.full(160,100)
delta_y_chip[79] = 200
delta_y_chip[80] = 200

delta_z = -0.01 * 10**6

# function to read a root file and return the needed information as a list of tuples (chip,column,row,weight)
def read_root(board1,board2,i):
    f_up = ROOT.TFile.Open('/home/patrick/PycharmProjects/Detektorpraktikum/detektor_praktikum/data/{}/measuring1/meas_{}.root'.format(board1,i),'read')
    f_down = ROOT.TFile.Open('/home/patrick/PycharmProjects/Detektorpraktikum/detektor_praktikum/data/{}/measuring1/meas_{}.root'.format(board2,i),'read')
    tree_up = f_up.Get('Xray/events')
    tree_down = f_down.Get('Xray/events')

    up_proc_array = rn.tree2array(tree_up, branches=['proc'])
    up_pcol_array = rn.tree2array(tree_up, branches=['pcol'])
    up_prow_array = rn.tree2array(tree_up, branches=['prow'])
    up_pq_array = rn.tree2array(tree_up, branches=['pq'])

    down_proc_array = rn.tree2array(tree_down, branches=['proc'])
    down_pcol_array = rn.tree2array(tree_down, branches=['pcol'])
    down_prow_array = rn.tree2array(tree_down, branches=['prow'])
    down_pq_array = rn.tree2array(tree_down, branches=['pq'])

    events_up = []
    events_down = []

    if len(up_proc_array) != len(down_proc_array):
        print('WARNING: Not the same number of events!')
        return events_up, events_down

    #if len(up_proc_array) != len(up_pcol_array) or len(up_proc_array) != len(up_prow_array) or len(up_proc_array) != len(up_pq_array):
    #    print('ups')
    #    return events


    for i in range(len(up_proc_array)):
        if up_proc_array[i][0] != [] and len(up_proc_array[i][0]) < 5 and down_proc_array[i][0] != [] and len(down_proc_array[i][0]) < 5:
            events_up.append((list(up_proc_array[i][0]),list(up_pcol_array[i][0]),list(up_prow_array[i][0]),list(up_pq_array[i][0])))
            events_down.append((list(down_proc_array[i][0]), list(down_pcol_array[i][0]), list(down_prow_array[i][0]),list(down_pq_array[i][0])))

    return events_up, events_down

# upper board = 1, lower = 2
def transform_into_xy(event,board=1):
    chips = np.array(event[0])
    cols = np.array(event[1])
    rows = np.array(event[2])
    weights = np.array(event[3])
    x_weighted = np.zeros(len(chips))
    y_weighted = np.zeros(len(chips))

    for i in range(len(chips)):
        chip = chips[i]
        col = cols[i]
        row = rows[i]
        if chip > 7:
            chip = 15 - chip
            col = 51 - col
            row = 79 + 80 - row
        # transform col into x
        x_weighted[i] = (chip * x_chip + np.sum(delta_x_chip[:col]) + 0.5 * delta_x_chip[col]) * weights[i]

        # transform row into y
        y_weighted[i] = (np.sum(delta_y_chip[:row]) + 0.5 * delta_y_chip[row]) * weights[i]

    # use weight to find center
    x = np.sum(x_weighted) / np.sum(weights)
    y = np.sum(y_weighted) / np.sum(weights)

    if board == 1:
        return TVector3(x,y,0)
    else:
        return TVector3(x,2*y_chip-y,delta_z)

def fit_2(hit_1, hit_2):
    s = hit_1
    t = hit_2-hit_1
    return [s + x * t for x in range(-5,5)]

for i in range(0,5):
    events_up, events_down = read_root('m4587','m4520',i)
    hits_up = []
    hits_down = []
    fits = []

    for i in range(len(events_up)):
    #for event in events_up:
        hit_up = transform_into_xy(events_up[i])
        hits_up.append(hit_up)

        hit_down = transform_into_xy(events_down[i],board=2)
        hits_down.append(hit_down)

        fits.append(fit_2(hit_up,hit_down))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot([x.X() for x in hits_up],[x.Y() for x in hits_up],[x.Z() for x in hits_up],'bs')
    ax.plot([x.X() for x in hits_down],[x.Y() for x in hits_down],[x.Z() for x in hits_down],'r*')
    for fit in fits:
        ax.plot([x.X() for x in fit], [x.Y() for x in fit], [x.Z() for x in fit], 'black')

    x_surf = np.arange(-0, 8*8100, 1000)
    y_surf = np.arange(-0, 2*8100, 1000)
    x_surf, y_surf = np.meshgrid(x_surf, y_surf)
    z_surf = np.zeros(x_surf.shape)
    ax.plot_surface(x_surf, y_surf, z_surf, color='b',alpha=0.4)
    ax.plot_surface(x_surf, y_surf, z_surf+delta_z, color='r', alpha=0.4)


    plt.ylim(ymax=-100, ymin=16300)
    plt.xlim(xmax=-100, xmin=8*8100+100)
    ax.set_zlim(delta_z-100,100)
    ax.grid(False)
    plt.show()

