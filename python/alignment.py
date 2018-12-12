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

x_fit_select = x_hist_upper[10000:20000]!=0
x_fit_hist = x_hist_upper[10000:20000]
popt, pcov = curve_fit(utils.gauss, np.array(range(10000,20000))[x_fit_select], x_fit_hist[x_fit_select], p0=[40000, 15000, 10000])
print(popt)
A = popt[0]
mu = popt[1]
sigma = popt[2]

ax.plot(np.array(range(0,X_SIZE))[x_upper_select], x_hist_upper[x_upper_select], 'b.', alpha=0.5, label="upper")
ax.plot(np.array(range(0,X_SIZE))[x_lower_select], x_hist_lower[x_lower_select], 'r.', alpha=0.5, label="lower")
ax.plot(range(0,X_SIZE), utils.gauss(range(0,X_SIZE), A, mu, sigma), 'k-', label="fit_upper")
ax.set_xlabel("x position in mu")


y_upper_select = y_hist_upper!=0
y_lower_select = y_hist_lower!=0
ay.plot(np.array(range(0,Y_SIZE))[y_upper_select], y_hist_upper[y_upper_select], 'b.', alpha=0.5, label="upper")
ay.plot(np.array(range(0,Y_SIZE))[y_upper_select], y_hist_lower[y_lower_select], 'r.', alpha=0.5, label="lower")
ay.set_xlabel("y position in mu")

plt.legend()
plt.show()

