import ROOT
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

upper_board = "m4587"
lower_board = "m4520"
lower_aligenment_raw_data = "../data/{}/alignment_1.root".format(lower_board)
upper_aligenment_raw_data = "../data/{}/alignment_1.root".format(upper_board)
#lower_aligenment_raw_data = "alignment_1.root"
#root_file = ROOT.TFile.Open(lower_aligenment_raw_data)

root_file = ROOT.TFile.Open(upper_aligenment_raw_data)
hist1 = root_file.Get('Xray/hMap_Ag_C7_V0')
#hist2 = root_file.Get('Xray/hMap_Ag_C8_V0')
canvas = ROOT.TCanvas()

hist1.Draw("COLZ")
#hist2.Draw("COLZ")
#canvas.Draw()

X_PIXEL = 52
Y_PIXEL = 80

x_hist  = np.zeros(X_PIXEL)
data_matrix = np.zeros((X_PIXEL, Y_PIXEL))
for x in range(1,X_PIXEL+1):
        for y in range(1,Y_PIXEL+1):

            if (y == 1 or y == Y_PIXEL) or (x == 1 or x == X_PIXEL):
                data_matrix[x-1][y-1] = hist1.GetCellContent(x, y)/2
                x_hist[x-1] += hist1.GetCellContent(x,y)/2
            else:
                data_matrix[x-1][y-1] = hist1.GetCellContent(x, y)
                x_hist[x-1] += hist1.GetCellContent(x, y)

print(data_matrix[30][50])

input("eingabe: ")
root_file.Close()

fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
#ax.bar3d(data_matrix)
ax.plot(range(0,X_PIXEL),x_hist)
plt.show()

