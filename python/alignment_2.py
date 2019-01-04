import ROOT
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import utils
from scipy.optimize import curve_fit

upper_board = "m4587"
lower_board = "m4520"
lower_aligenment_raw_data = "../data/{}/alignment_2.root".format(lower_board)
upper_aligenment_raw_data = "../data/{}/alignment_2.root".format(upper_board)
#lower_aligenment_raw_data = "alignment_1.root"
#root_file = ROOT.TFile.Open(lower_aligenment_raw_data)

root_file = ROOT.TFile.Open(upper_aligenment_raw_data)

	
#hist2 = root_file.Get('Xray/hMap_Ag_C8_V0')
#canvas = ROOT.TCanvas()

#hist1.Draw("COLZ")
#hist2.Draw("COLZ")
#canvas.Draw()

X_PIXEL = 52
Y_PIXEL = 80
X_SIZE = int(8*8100)
Y_SIZE = int(2*8100)

x_hist_upper = np.zeros(X_SIZE)
y_hist_upper = np.zeros(Y_SIZE)

x_hist_lower = np.zeros(X_SIZE)
y_hist_lower = np.zeros(Y_SIZE)

data_matrix = np.zeros((X_SIZE, Y_SIZE))

with utils.ShutUpRoot():
	for i in range (0,16):
		hist_name = 'Xray/hMap_Ag_C{}_V0'.format(i)
		print("load hist: {}".format(hist_name))
		root_hist = root_file.Get(hist_name)
		for x in range(1,X_PIXEL+1):
			for y in range(1,Y_PIXEL+1):
	
				event = utils.transform_into_xy(np.array([[i],[x-1],[y-1],[1]]), 1)
				if (y == 1 or y == Y_PIXEL) or (x == 1 or x == X_PIXEL):
	#data_matrix[x-1][y-1] = hist1.GetCellContent(x, y)/2
					x_hist_upper[int(event[0])] += root_hist.GetCellContent(x,y)/2
					y_hist_upper[int(event[1])] += root_hist.GetCellContent(x,y)/2
				
				else:
	#      	data_matrix[x-1][y-1] = hist1.GetCellContent(x, y)
					x_hist_upper[int(event[0])] += root_hist.GetCellContent(x, y)
					y_hist_upper[int(event[1])] += root_hist.GetCellContent(x, y)
	
	
	root_file = ROOT.TFile.Open(lower_aligenment_raw_data)
	
	for i in range (0,16):
		hist_name = 'Xray/hMap_Ag_C{}_V0'.format(i)
		print("load hist: {}".format(hist_name))
		root_hist = root_file.Get(hist_name)
		for x in range(1,X_PIXEL+1):
			for y in range(1,Y_PIXEL+1):
	
				event = utils.transform_into_xy(np.array([[i],[x-1],[y-1],[1]]), 2)
				if (y == 1 or y == Y_PIXEL) or (x == 1 or x == X_PIXEL):
	#data_matrix[x-1][y-1] = hist1.GetCellContent(x, y)/2
					x_hist_lower[int(event[0])] += root_hist.GetCellContent(x,y)/2
					y_hist_lower[int(event[1])] += root_hist.GetCellContent(x,y)/2
				
				else:
	#      	data_matrix[x-1][y-1] = hist1.GetCellContent(x, y)
					x_hist_lower[int(event[0])] += root_hist.GetCellContent(x, y)
					y_hist_lower[int(event[1])] += root_hist.GetCellContent(x, y)
	
	
#print(data_matrix[30][50])

#input("eingabe: ")
root_file.Close()

fig = plt.figure()
ax = fig.add_subplot(121)#, projection='3d')
ay = fig.add_subplot(122)
#ax.bar3d(data_matrix)
x_upper_select = x_hist_upper!=0
x_lower_select = x_hist_lower!=0
y_upper_select = y_hist_upper!=0
y_lower_select = y_hist_lower!=0



# Fit environment for x direction
x_min = 0
x_max = 30000

# upper board
x_fit_select = x_hist_upper[x_min:x_max]!= 0 # filter out bins where zero entries
x_fit_hist = x_hist_upper[x_min:x_max] # histogramm contains just the bins with entries
popt, pcov = curve_fit(utils.gauss, np.array(range(x_min,x_max))[x_fit_select], x_fit_hist[x_fit_select], p0=[40000, 15000, 10000])
print("x upper board:")
print(popt)
A_up = popt[0]
mu_up = popt[1]
sigma_up = popt[2]

# lower board
x_fit_select = x_hist_lower[x_min:x_max]!= 0 # filter out bins where zero entries
x_fit_hist = x_hist_lower[x_min:x_max] # histogramm contains just the bins with entries
popt, pcov = curve_fit(utils.gauss, np.array(range(x_min,x_max))[x_fit_select], x_fit_hist[x_fit_select], p0=[40000, 15000, 10000])
print("x lower board:")
print(popt)
A_low = popt[0]
mu_low = popt[1]
sigma_low = popt[2]




# Fit environment for y direction
y_min = 0
y_max = Y_SIZE

#upper board
y_fit_select = y_hist_upper[y_min:y_max]!= 0
y_fit_hist = y_hist_upper[y_min:y_max]
popt, pcov = curve_fit(utils.gauss, np.array(range(y_min,y_max))[y_fit_select], y_fit_hist[y_fit_select], p0=[25000, 8000, 10000])
print("y upper board:")
print(popt)
A_yup = popt[0]
mu_yup = popt[1]
sigma_yup = popt[2]

#lower board
y_fit_select = y_hist_lower[y_min:y_max]!= 0
y_fit_hist = y_hist_lower[y_min:y_max]
popt, pcov = curve_fit(utils.gauss, np.array(range(y_min,y_max))[y_fit_select], y_fit_hist[y_fit_select], p0=[8000, 8000, 10000])
print("y lower board:")
print(popt)
A_ylow = popt[0]
mu_ylow = popt[1]
sigma_ylow = popt[2]




ax.plot(np.array(range(0,X_SIZE))[x_upper_select]/1000, x_hist_upper[x_upper_select], 'b.', alpha=0.5, label="upper")
ax.plot(np.array(range(0,X_SIZE))[x_lower_select]/1000, x_hist_lower[x_lower_select], 'r.', alpha=0.5, label="lower")
ax.plot(np.arange(0,X_SIZE)/1000, utils.gauss(range(0,X_SIZE), A_up, mu_up, sigma_up), 'b-', label="fit_upper")
ax.plot(np.arange(0,X_SIZE)/1000, utils.gauss(range(0,X_SIZE), A_low, mu_low, sigma_low), 'r-', label="fit_upper")
ax.set_xlabel("x position in mm")
ax.set_ylabel("intensity in arbitrary unit")



ay.plot(np.array(range(0,Y_SIZE))[y_upper_select]/1000, y_hist_upper[y_upper_select], 'b.', alpha=0.5, label="upper")
ay.plot(np.array(range(0,Y_SIZE))[y_upper_select]/1000, y_hist_lower[y_lower_select], 'r.', alpha=0.5, label="lower")
ay.plot(np.arange(0,Y_SIZE)/1000, utils.gauss(range(0,Y_SIZE), A_yup, mu_yup, sigma_yup), 'b-', label="fit_upper")
ay.plot(np.arange(0,Y_SIZE)/1000, utils.gauss(range(0,Y_SIZE), A_ylow, mu_ylow, sigma_ylow), 'r-', label="fit_lower")
ay.set_xlabel("y position in mm")

plt.legend()
plt.show()

