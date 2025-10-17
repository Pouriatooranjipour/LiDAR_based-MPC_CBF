from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
from simulation_code import simulate
import matplotlib.pyplot as plt

# setting matrix_weights' variables
Q_x = 10
Q_y = 1
Q_theta = 2
R1 = 1
R2 = 0.5

gamma = 0.2
# Obstacle parameters
o_x = -2  # Obstacle center x-coordinate
o_y = -2  # Obstacle center y-coordinate
obstacle_radius = 1  # Radius of the obstacle


step_horizon = 0.1  # time between steps in seconds
N = 6             # number of look ahead steps
rob_diam = 0.3      # diameter of the robot
sim_time = 30      # simulation time

# specs
x_init = 0
y_init = 0
theta_init = -pi
x_target = -3
y_target = -4
theta_target = -pi/2

v_max = 1
v_min = -1


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()


# control symbolic variables
v = ca.SX.sym('v')
omega = ca.SX.sym('omega')
controls = ca.vertcat(v, omega)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

hh_next = ca.DM.zeros(N ,1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)

# controls weights matrix
R = ca.diagcat(R1, R2)

# discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
RHS = ca.vertcat(v*cos(theta), v*sin(theta), omega)  # system r.h.s
# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation
safety_distance_squared = (obstacle_radius + rob_diam/2)**2
G = []

# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)
    distance_to_obstacle_squared = (X[0, k] - o_x)**2 + (X[1, k] - o_y)**2
    distance_to_obstacle_squared_t = (st_next[0] - o_x)**2 + (st_next[1] - o_y)**2
    cbf_constraint = distance_to_obstacle_squared - safety_distance_squared
    cbf_constraint_t = distance_to_obstacle_squared_t - safety_distance_squared
    G = ca.vertcat(G, cbf_constraint_t + (gamma-1) * cbf_constraint)

g = ca.vertcat(g, G)

OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf     # X lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # X upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

lbx[n_states*(N+1):] = v_min                  # v lower bound for all V
ubx[n_states*(N+1):] = v_max                  # v upper bound for all V

lbg_cbf = ca.DM.ones(N) * 0.1  # This ensures a safety distance is maintained
ubg_cbf = ca.DM.inf(N)  # Upper bounds for the CBF constraints for each timestep

# Update the upper bounds for the constraints to include CBF constraints
# Assuming the previous size of ubg was n_states*(N+1)


# Update the lower bounds for the constraints to include CBF constraints
# Assuming the previous size of lbg was n_states*(N+1)

args = {
    'lbx': lbx,
    'ubx': ubx
}
args['lbg'] = ca.vertcat(
    ca.DM.zeros((n_states*(N+1), 1)),  # Original lower bounds
    lbg_cbf  # Lower bounds for the CBF constraints
)

args['ubg'] = ca.vertcat(
    ca.DM.zeros((n_states*(N+1), 1)),  # Original upper bounds for state constraints
    ubg_cbf  # Upper bounds for the CBF constraints
)

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_target = ca.DM([x_target, y_target, theta_target])  # target state

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])
X = np.array([x_init , y_init])

###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    while (ca.norm_2(state_init - state_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f)
        X = np.vstack((X, state_init[0:2].T))
        # print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        print(mpc_iter)
        print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    # simulate(cat_states, cat_controls, times, step_horizon, N,
    #          np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), o_x, o_y, obstacle_radius, save=True)

    # np.savez('robot_states_N8.npz', x_states=X[:,0], y_states=X[:,1])