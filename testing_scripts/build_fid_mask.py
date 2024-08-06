import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define the centre
centre = np.array([0, 0])
sc = 3.0

# define the radius of the first circle
r1 = 5.0
angles1 = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 7*np.pi/4])
pos1 = np.array([r1*np.cos(angles1), r1*np.sin(angles1)]).T + centre
s1 = 3.0 * np.ones_like(angles1)

# # # second circle
# r2 = 7.24
# angles2 = np.arange(np.pi/8, 2*np.pi, np.pi/4)
# pos2 = np.array([r2*np.cos(angles2), r2*np.sin(angles2)]).T + centre
# s2 = 2.0 * np.ones_like(angles2)

# # line
# line = np.array([[0, -4], [0, -7]])
# s3 = np.array([2, 2])

line = np.array([[0, -4]])
s3 = np.array([2])

# create an array of all points
# all_points = np.vstack((centre, pos1, pos2, line))
all_points = np.vstack((centre, pos1, line))

# create a data frame of the points
df = pd.DataFrame(all_points, columns=['y', 'z'])
# df['s'] = np.hstack((sc, s1, s2, s3))
df['s'] = np.hstack((sc, s1, s3))
# df['x'] = np.zeros_like(df['y'])
df.to_csv('data/PAE/metro_accuracy/fid_as_built.csv', index=False)

# plot the first circle
fig, ax = plt.subplots()
ax.scatter(pos1[:, 0], pos1[:, 1], label='Circle 1', s=s1*100)
# ax.scatter(pos2[:, 0], pos2[:, 1], label='Circle 2', s=s2*100)
ax.scatter(line[:, 0], line[:, 1], label='Line', s=s3*100)
ax.scatter(centre[0], centre[1], label='Centre', color='r', s=sc*100)
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.legend(markerscale=0.5)
ax.grid()
plt.show()