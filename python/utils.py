import numpy as np
import ROOT
from ROOT import TVector3
import root_numpy as rn

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

# upper board = 1, lower = 2
def transform_into_xy(event,board):
		# event[0]: chip number
		# event[1]: colum (on chip)
		# event[2]: row (on chip)
		# event[3]: weight

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

# function to read a root file and return the needed information as a list of tuples (chip,column,row,weight)
def read_root_measurement(board1,board2,i):
    f_up = ROOT.TFile.Open('data/{}/measuring1/meas_{}.root'.format(board1,i),'read')
    f_down = ROOT.TFile.Open('data/{}/measuring1/meas_{}.root'.format(board2,i),'read')
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

    for i in range(len(up_proc_array)):
        if up_proc_array[i][0] != [] and len(up_proc_array[i][0]) < 5 and down_proc_array[i][0] != [] and len(down_proc_array[i][0]) < 5:
            events_up.append((list(up_proc_array[i][0]),list(up_pcol_array[i][0]),list(up_prow_array[i][0]),list(up_pq_array[i][0])))
            events_down.append((list(down_proc_array[i][0]), list(down_pcol_array[i][0]), list(down_prow_array[i][0]),list(down_pq_array[i][0])))

    return events_up, events_down

# function to read an alignemnt file and return the single events
def read_root_alignment(file):
    f = ROOT.TFile.Open(file,'read')
    tree = f.Get('Xray/events')

    proc_array = rn.tree2array(tree, branches=['proc'])
    pcol_array = rn.tree2array(tree, branches=['pcol'])
    prow_array = rn.tree2array(tree, branches=['prow'])
    pq_array = rn.tree2array(tree, branches=['pq'])

    events = []

    for i in range(len(proc_array)):
        if proc_array[i][0] != [] and len(proc_array[i][0]) < 5:
            events.append((list(proc_array[i][0]),list(pcol_array[i][0]),list(prow_array[i][0]),list(pq_array[i][0])))

    return events




class ShutUpRoot():
    """Context manager for silencing certain ROOT operations.  Usage:
    with Quiet(level = ROOT.kInfo+1):
       foo_that_makes_output

    You can set a higher or lower warning level to ignore different
    kinds of messages.  After the end of indentation, the level is set
    back to what it was previously.
    """
    def __init__(self, level=ROOT.kError):
        self.level = level

    def __enter__(self):
        self.oldlevel = ROOT.gErrorIgnoreLevel
        ROOT.gErrorIgnoreLevel = self.level

    def __exit__(self, type, value, traceback):
        ROOT.gErrorIgnoreLevel = self.oldlevel


def gauss(x, A, mu, sigma):
	return A*np.exp(-(x-mu)**2/(2.*sigma**2)) 
