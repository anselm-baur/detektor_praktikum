import numpy as np
import ROOT
from ROOT import TVector3
import root_numpy as rn
import matplotlib.pyplot as plt

# define some global variables values in µm
x_chip = 8100
y_chip = 8100
# chip
delta_x_chip = np.full(52,150)
delta_x_chip[0] = 300
delta_x_chip[51] = 300

delta_y_chip = np.full(160,100)
delta_y_chip[0] = 200
delta_y_chip[79] = 200
delta_y_chip[80] = 200
delta_y_chip[159] = 200

delta_z = 0.01 * 10**6

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
        return TVector3(x,y,delta_z)
    else:
        return TVector3(x,2*y_chip-y,0)


# transfers the pixel with chip number to a coordinate in mu
def px_to_mu(roc, col, row, board):
    if roc > 7:
        roc = 15 - roc
        col = 51 - col
        row = 79 + 80 - row

    x = roc * x_chip + np.sum(delta_x_chip[:col]) + 0.5 * delta_x_chip[col]
    y = np.sum(delta_y_chip[:row]) + 0.5 * delta_y_chip[row]

    if board == "upper":
        return [x,y]
    elif board == "lower":
        return [x,2*y_chip-y]



# this function is for scrpts in the python folder of detector_practikum directory
def load_root_measurement(campaign,board1,board2,measurement):
    f_up = ROOT.TFile.Open('../data/{}/measuring{}/meas_{}.root'.format(board1, campaign, measurement), 'read')
    f_down = ROOT.TFile.Open('../data/{}/measuring{}/meas_{}.root'.format(board2, campaign, measurement), 'read')
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

    # if len(up_proc_array) != len(up_pcol_array) or len(up_proc_array) != len(up_prow_array) or len(up_proc_array) != len(up_pq_array):
    #    print('ups')
    #    return events

    for i in range(len(up_proc_array)):
        if up_proc_array[i][0] != [] and len(up_proc_array[i][0]) < 5 and down_proc_array[i][0] != [] and len(
            down_proc_array[i][0]) < 5:
            events_up.append((list(up_proc_array[i][0]), list(up_pcol_array[i][0]), list(up_prow_array[i][0]),
                              list(up_pq_array[i][0])))
            events_down.append((list(down_proc_array[i][0]), list(down_pcol_array[i][0]), list(down_prow_array[i][0]),
                                list(down_pq_array[i][0])))

    return events_up, events_down


# testing around with raw pixels
def get_pixel_hits(campaign, measurement):
    board_name = {"upper":"m4587", "lower":"m4520"}
    f_up = ROOT.TFile.Open('../data/{}/measuring{}/meas_{}.root'.format(board_name["upper"],campaign,measurement),'read')
    f_down = ROOT.TFile.Open('../data/{}/measuring{}/meas_{}.root'.format(board_name["lower"],campaign,measurement),'read')

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

    x_hits_up = []
    y_hits_up = []

    x_hits_down = []
    y_hits_down = []

    #for i in range(len(proc_array)):
    #print("proc_array: {}".format(list(proc_array[event][0]).__len__()))
    #print("pcol_array: {}".format(list(pcol_array[event][0])))
    #print("prow_array: {}".format(list(prow_array[event][0])))
    #print("pq_array: {}".format(list(pq_array[event][0])))
    #print(list(pcol_array))

    num = 0
    for i in range(len(up_proc_array)):
        if up_proc_array[i][0] != [] and len(up_proc_array[i][0]) < 5 and down_proc_array[i][0] != [] and len(down_proc_array[i][0]) < 5:
            print(i)
            up_procs = list(up_proc_array[i][0])
            up_pcols = list(up_pcol_array[i][0])
            up_prows = list(up_prow_array[i][0])

            down_procs = list(down_proc_array[i][0])
            down_pcols = list(down_pcol_array[i][0])
            down_prows = list(down_prow_array[i][0])

            #print(list(proc_array[i][0]))
            #print(list(proc_array[i][0])[0])
            #print(list(pcol_array[i][0]))
            #print(list(pcol_array[i][0])[0])
            #print(list(prow_array[i][0]))

            #print(num)
            #print("--------------------")

            for j in range(len(up_procs)):
                [x,y] = px_to_mu(up_procs[j], up_pcols[j], up_prows[j], "uppwer")
                x_hits_up.append(x)
                y_hits_up.append(y)
            for j in range(len(down_procs)):
                [x,y] = px_to_mu(down_procs[j], down_pcols[j], down_prows[j], "uppwer")
                x_hits_down.append(x)
                y_hits_down.append(y)
        num += 1
    #print(num)
    return [x_hits_up, y_hits_up, x_hits_down, y_hits_down]



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


def align_theta():
    # values in mu meter
    x_a1 = 52885  # x upper pos 1
    x_a2 = 15229  # x upper pos 2
    x_b1 = 52850  # x lower pos 1 not aligned
    x_b2 = 15068  # x lower pos 2 not aligned

    y_a1 = 8655
    y_a2 = 7656
    y_b1 = 8138
    y_b2 = 7942

    upper_vec = TVector3((x_a1-x_a2),(y_a1-y_a2),0)
    lower_vec = TVector3((x_b1-x_b2),(y_b1-y_b2),0)

    #fig = plt.figure()

    #plt.plot([0,upper_vec.X()/1000], [0,upper_vec.Y()/1000])
    #plt.plot([0,lower_vec.X()/1000], [0,lower_vec.Y()/1000])
    #plt.show()

    cos_theta = ((x_a1 - x_a2) * (x_b1 - x_b2) + (y_a1 - y_a2) * (y_b1 - y_b2)) / \
                np.sqrt(((x_a1 - x_a2) ** 2 + (x_b1 - x_b2) ** 2) * ((y_a1 - y_a2) ** 2 + (y_b1 - y_b2) ** 2))

    #print("theta: {}".format(upper_vec.Angle(lower_vec)))

    return upper_vec.Angle(lower_vec)



def align_shift():
    # values in mu meter
    x_a1 = 52885  # x upper pos 1
    x_a2 = 15229  # x upper pos 2
    x_b1 = 52850  # x lower pos 1 not aligned
    x_b2 = 15068  # x lower pos 2 not aligned

    y_a1 = 8655
    y_a2 = 7656
    y_b1 = 8138
    y_b2 = 7942

    theta = align_theta()

    p1 = TVector3(x_b1, y_b1, 0)
    p2 = TVector3(x_b2, y_b2, 0)

    vec_p1 = np.matrix((p1.X(), p1.Y()))
    vec_p2 = np.matrix((p2.X(), p2.Y()))
    rot_matrix = np.matrix(((np.cos(theta), -np.sin(theta)),
                              (np.sin(theta), np.cos(theta))))

    vec_p1 = rot_matrix * vec_p1.transpose()
    vec_p2 = rot_matrix * vec_p2.transpose()

    p1 = TVector3(vec_p1.item(0,0),vec_p1.item(1,0),0)
    p2 = TVector3(vec_p2.item(0, 0), vec_p2.item(1, 0), 0)

    d1 = TVector3(x_a1,y_a1,0) - p1
    d2 = TVector3(x_a2, y_a2, 0) - p2

    dx = (d1.X() + d2.X()) / 2
    dy = (d1.Y() + d2.Y()) / 2

    #print("dx_1: {}, dy_1: {}".format(d1.X(), d1.Y()))
    #print("dx_2: {}, dy_2: {}".format(d2.X(), d2.Y()))

    #dx = x_a1 - x_b1*np.cos(theta) + y_b1*np.sin(theta)
    #dy = y_a1 - x_b1*np.sin(theta) - y_b1*np.cos(theta)

    return [dx, dy]


def board_alignment(hit_position):
    dx, dy = align_shift()
    theta = align_theta()

    #print("theta: {}; dx: {}; dy: {}".format(theta, dx, dy))

    vec = np.matrix((hit_position.X(), hit_position.Y(),1))
    align_matrix = np.matrix(((np.cos(theta),-np.sin(theta), dx),
                              (np.sin(theta), np.cos(theta), dy),
                              (0,0,1)))

    new_vector = align_matrix*vec.transpose()
    #print(new_vector)

    return TVector3(new_vector.item(0,0), new_vector.item(1,0), hit_position.Z())




def fit_trace(hit_1, hit_2):
    s = hit_1
    t = hit_2 - hit_1

    v = []
    for x in range(-5, 5):
        u = TVector3(t.X() * x, t.Y() * x, t.Z() * x)
        v.append(s + u)

    return v


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
