from utils import *


fig = plt.figure()
ax1 = fig.add_subplot(111)
#ax2 = fig.add_subplot(122)

hits_up = []
hits_down = []

x_hits = []
y_hits = []

num_measurements = [50,0]
for campaign in range(1,2):
    for i in range(num_measurements[campaign-1],51):
        events_up, events_down = load_root_measurement(campaign, 'm4587', 'm4520', i)
        print("File: {}/meas_{}, Events: {}/{}".format(campaign, i, len(events_up), len(events_down)))

        [x_hits_u, y_hits_u, x_hits_d, y_hits_d] = get_pixel_hits(campaign, i)


        for i in range(len(events_up)):
            hit_up = transform_into_xy(events_up[i], board=1)
            hit_down = transform_into_xy(events_down[i], board=2)
            hits_up.append(hit_up)
            hits_down.append(hit_down)


ax1.plot([x for x in x_hits_u], [y for y in y_hits_u], "b*", alpha=1)
ax1.plot([x for x in x_hits_d], [y for y in y_hits_d], "r*", alpha=1)
ax1.plot([x.X() for x in hits_up], [y.Y() for y in hits_up], "bo", alpha=0.5)
ax1.plot([x.X() for x in hits_down], [y.Y() for y in hits_down], "ro", alpha=0.5)

plt.show()