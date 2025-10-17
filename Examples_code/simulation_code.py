import numpy as np
from numpy import sin, cos, pi
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from time import time

def simulate(cat_states, cat_controls, t, step_horizon, N, reference, o_x, o_y, obstacle_radius, save=False):
    def create_triangle(state=[0,0,0], h=0.15, w=0.3, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [cos(th), -sin(th)],
            [sin(th),  cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        return path, horizon, current_state, target_state,

    def animate(i):
        x = cat_states[0, 0, i]
        y = cat_states[1, 0, i]
        th = cat_states[2, 0, i]

        # Update path with the current position
        if i < len(t) - 1:  # Extend path if not the last frame
            x_new = np.hstack((path.get_xdata(), x))
            y_new = np.hstack((path.get_ydata(), y))
        else:  # For the last frame, do not extend the path
            x_new = path.get_xdata()
            y_new = path.get_ydata()

        path.set_data(x_new, y_new)

        # Update horizon with all positions in this frame
        x_new = cat_states[0, :, i]
        y_new = cat_states[1, :, i]
        horizon.set_data(x_new, y_new)

        # Update current state triangle position
        current_state.set_xy(create_triangle([x, y, th], update=True))

        # Update target state triangle position
        target_x = reference[3]  # Replace with actual target x position
        target_y = reference[4]  # Replace with actual target y position
        target_th = reference[5]  # Replace with actual target orientation
        target_state.set_xy(create_triangle([target_x, target_y, target_th], update=True))

        return path, horizon, current_state, target_state,


    fig, ax = plt.subplots(figsize=(6, 6))
    min_scale = min(reference[0], reference[1], reference[3], reference[4], o_x - obstacle_radius - 1, o_y - obstacle_radius - 1) - 2
    max_scale = max(reference[0], reference[1], reference[3], reference[4], o_x + obstacle_radius + 1, o_y + obstacle_radius + 1) + 2
    ax.set_xlim(left=min_scale, right=max_scale)
    ax.set_ylim(bottom=min_scale, top=max_scale)

    # Draw obstacle with red color
    obstacle = Circle((o_x, o_y), obstacle_radius, color='red', alpha=0.5, label='Obstacle')
    ax.add_patch(obstacle)

    path, = ax.plot([], [], 'k', linewidth=2)
    horizon, = ax.plot([], [], 'x-g', alpha=0.5)

    current_triangle = create_triangle(reference[:3])
    current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color='r')[0]

    target_triangle = create_triangle(reference[3:])
    target_state = ax.fill(target_triangle[:, 0], target_triangle[:, 1], color='b')[0]

    anim = animation.FuncAnimation(
        fig=fig,
        func=animate,
        init_func=init,
        frames=len(t),
        interval=step_horizon*1000,
        blit=False,
        repeat=True
    )
    plt.show()

    if save:
        anim.save('animation' + str(time()) + '.gif', writer='imagemagick', fps=15)
