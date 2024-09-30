import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from time import time

o_x = -2  # Obstacle center x-coordinate
o_y = -2  # Obstacle center y-coordinate
obstacle_radius = 1  # Radius of the obstacle

data = np.load('robot_states_N6_gamma_02.npz')
x_states = data['x_states']
y_states = data['y_states']

plt.plot(x_states, y_states, label='L=6, gamma = 0.2', marker='o', linestyle='-')

data = np.load('robot_states_N8.npz')
x_states = data['x_states']
y_states = data['y_states']

plt.plot(x_states, y_states, label='L=8', marker='o', linestyle='-')

data = np.load('robot_states_N6.npz')
x_states = data['x_states']
y_states = data['y_states']

plt.plot(x_states, y_states, label='L=6', marker='o', linestyle='-')


# Plot the obstacle as a circle
obstacle_circle = plt.Circle((o_x, o_y), obstacle_radius, color='r', alpha=0.5, label='Obstacle')
plt.gca().add_patch(obstacle_circle)

# Setting labels and title
plt.xlabel('x(m)',fontsize=12, fontweight='bold')
plt.ylabel('y(m)',fontsize=12, fontweight='bold')
# plt.title('Obstacle avoidance in MPC',fontsize=14, fontweight='bold')
plt.legend(fontsize=10, frameon=True)
plt.grid(True)
plt.axis('equal')

plt.savefig('robot_trajectories_MPC_gamma.eps', format='eps')
# Show the plot
plt.show()
