import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion
import casadi as ca
from casadi import sin, cos, pi
from std_msgs.msg import Float64MultiArray
# import time


class State:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

class Input:
    def __init__(self, v=0.0, w=0.0):
        self.v = v
        self.w = w

class MPCCBFNode(Node):
    def __init__(self):
        super().__init__('MPC_CBF')
        self.get_clock().use_sim_time = True
        # Subscribe to the filtered odometry data for accurate pose estimation
        self.odom_subscription = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.obst_subscription = self.create_subscription(
            Float64MultiArray,
            '/lidar_ellipses',
            self.ellipse_callback,
            10)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        if self.obst_subscription is None:
            self.get_logger().error('Failed to subscribe to /lidar_ellipses')
        else:
            self.get_logger().info('Successfully subscribed to /lidar_ellipses')
        
        # self.o_x = 1.5
        # self.o_y = 0
        # self.obstacle_radius = 0.1
        self.gamma = 0.5
        self.ellipse_recieved = True

        self.obstacles = []  # Initialize an empty list to store ellipses


        
        self.dt_ = 0.1
        self.N = 20
        self.rob_diam = 0

        self.Q_ = ca.diagcat(0.3, 0.02, 0.04)
        self.R_ = ca.diagcat(0.3, 0.3)

        self.desired_state_ = State(3, 0, pi/2)
        self.max_linear_velocity = 1.0
        self.max_angular_velocity = pi/2

        self.tolerance = 0.1
        self.end_controller = False

        self.actual_state_ = State()
        
            
        self.X0 = ca.DM.zeros((3, self.N+1))         # initial state full

        self.u0 = ca.DM.zeros((2, self.N))  # initial control
        self.control_loop_timer_ = self.create_timer(self.dt_, self.control_loop_callback)

    def odom_callback(self, msg):
        # Extract the robot's current pose from the Odometry message
        position = msg.pose.pose.position
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (_, _, yaw) = euler_from_quaternion(orientation_list)
        self.actual_state_ = State(position.x, position.y, yaw)

    def ellipse_callback(self, msg):
        # Process the received ellipse data
        self.obstacles.clear()  # Clear the list to update it with new data

        data = msg.data
        # Extract ellipses from the received data
        for i in range(0, len(data), 4):
            x_c = data[i]
            y_c = data[i + 1]
            r_x = data[i + 2]
            r_y = data[i + 3]
            self.obstacles.append((x_c, y_c, r_x, r_y))
        
        # self.o_x = x_c
        # self.o_y = y_c
        # self.obstacle_radius = (r_x + r_y)/2
        # Here, you can use the ellipses as needed in your node
        self.get_logger().info(f'Received {len(self.obstacles)} ellipses: {self.obstacles}')
        if len(self.obstacles) != 0:
            self.ellipse_received = True  # Set the flag to True

    def combined_cbf_constraint(self, x, y):
        # Initialize log-sum-exp value
        lse_value = 0
        k = 10
        # Compute the exponentials for each ellipse
        for (o_x, o_y, r_x, r_y) in self.obstacles:
            # Calculate individual h_i for each obstacle
            h_i = (x - o_x)**2 + (y - o_y)**2 - ((r_x + r_y) / 2)**2
            
            # Accumulate the exponentials for the LSE function
            lse_value += ca.exp(-k * h_i)
        
        # Calculate the combined CBF constraint h using Log-Sum-Exp
        combined_h = -(1 / k) * ca.log(lse_value)
        return combined_h

    def MPC_CBF_Control(self, state_init, state_target):
        # MPC Optimization
        x = ca.SX.sym('x')
        y = ca.SX.sym('y')
        theta = ca.SX.sym('theta')
        states = ca.vertcat(
            x,
            y,
            theta
        )
        v = ca.SX.sym('v')
        omega = ca.SX.sym('omega')
        controls = ca.vertcat(v, omega)
        X = ca.SX.sym('X', 3, self.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        U = ca.SX.sym('U', 2, self.N)

        # coloumn vector for storing initial state and target state
        P = ca.SX.sym('P', 6)

        # discretization model (e.g. x2 = f(x1, v, t) = x1 + v * dt)
        RHS = ca.vertcat(v*cos(theta), v*sin(theta), omega)  # system r.h.s
        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
        f = ca.Function('f', [states, controls], [RHS])


        cost_fn = 0  # cost function
        g = X[:, 0] - P[:3]  # constraints in the equation
        G = []

        for k in range(self.N):
            st = X[:, k]
            con = U[:, k]
            cost_fn = cost_fn \
                + (st - P[3:]).T @ self.Q_ @ (st - P[3:]) \
                + con.T @ self.R_ @ con
            st_next = X[:, k+1]
            k1 = f(st, con)
            k2 = f(st + self.dt_/2*k1, con)
            k3 = f(st + self.dt_/2*k2, con)
            k4 = f(st + self.dt_ * k3, con)
            st_next_RK4 = st + (self.dt_ / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g = ca.vertcat(g, st_next - st_next_RK4)

            if self.ellipse_recieved:
                combined_h = self.combined_cbf_constraint(X[0, k], X[1, k])
                combined_h_t = self.combined_cbf_constraint(st_next[0], st_next[1])
                
                # CBF constraint
                cbf_constraint = combined_h
                cbf_constraint_t = combined_h_t
                
                # Combine the CBF constraints
                G = ca.vertcat(G, cbf_constraint_t + (self.gamma - 1) * cbf_constraint)


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

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        lbx = ca.DM.zeros((3*(self.N+1) + 2*self.N, 1))
        ubx = ca.DM.zeros((3*(self.N+1) + 2*self.N, 1))

        lbx[0: 3*(self.N+1): 3] = -ca.inf     # X lower bound
        lbx[1: 3*(self.N+1): 3] = -ca.inf     # Y lower bound
        lbx[2: 3*(self.N+1): 3] = -ca.inf     # theta lower bound

        ubx[0: 3*(self.N+1): 3] = ca.inf      # X upper bound
        ubx[1: 3*(self.N+1): 3] = ca.inf      # Y upper bound
        ubx[2: 3*(self.N+1): 3] = ca.inf      # theta upper bound

        lbx[3*(self.N+1):] = -5                  # v lower bound for all V
        ubx[3*(self.N+1):] = 5                  # v upper bound for all V

        lbg_cbf = ca.DM.ones(self.N) * 0.1  # This ensures a safety distance is maintained
        ubg_cbf = ca.DM.inf(self.N)  # Upper bounds for the CBF constraints for each timestep

        # Update the upper bounds for the constraints to include CBF constraints
        # Assuming the previous size of ubg was n_states*(N+1)


        # Update the lower bounds for the constraints to include CBF constraints
        # Assuming the previous size of lbg was n_states*(N+1)

        self.args = {
            'lbx': lbx,
            'ubx': ubx
        }
        if self.ellipse_recieved:
            self.args['lbg'] = ca.vertcat(
                ca.DM.zeros((3*(self.N+1), 1)),  # Original lower bounds
                lbg_cbf  # Lower bounds for the CBF constraints
            )
        else:
            self.args['lbg'] = ca.vertcat(
                ca.DM.zeros((3*(self.N+1), 1))
            )
        
        if self.ellipse_recieved:
            self.args['ubg'] = ca.vertcat(
                ca.DM.zeros((3*(self.N+1), 1)),  # Original upper bounds for state constraints
                ubg_cbf  # Upper bounds for the CBF constraints
            )
        else:
            self.args['ubg'] = ca.vertcat(
                ca.DM.zeros((3*(self.N+1), 1))
            )
        
        
        
        
        self.args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        self.args['x0'] = ca.vertcat(
            ca.reshape(self.X0, 3*(self.N+1), 1),
            ca.reshape(self.u0, 2*self.N, 1)
        )
        sol = self.solver(
            x0=self.args['x0'],
            lbx=self.args['lbx'],
            ubx=self.args['ubx'],
            lbg=self.args['lbg'],
            ubg=self.args['ubg'],
            p=self.args['p']
        )
        u = ca.reshape(sol['x'][3 * (self.N + 1):], 2, self.N)
        self.u0 = u
        self.X0 = ca.reshape(sol['x'][: 3 * (self.N+1)], 3, self.N+1)
        return np.array([float(u[0, 0]), float(u[1, 0])])
    
    def control_loop_callback(self):

        # start_time = time.time()  # Start timing

        if not self.end_controller:
            x_actual = np.array([self.actual_state_.x, self.actual_state_.y, self.actual_state_.theta])
            x_desired = np.array([self.desired_state_.x, self.desired_state_.y, self.desired_state_.theta])
            state_error = x_actual - x_desired

            u = self.MPC_CBF_Control(x_actual, x_desired)

            vel_msg = Twist()
            vel_msg.linear.x = u[0]
            vel_msg.angular.z = u[1]
            self.publisher.publish(vel_msg)

            # end_time = time.time()  # End timing

            # computation_time = end_time - start_time
            # self.get_logger().info(f'Computation Time: {computation_time} seconds')

            if np.linalg.norm(state_error) < self.tolerance:
                self.end_controller = True
                self.get_logger().info('Goal reached!')
                self.publisher.publish(Twist())  # Stop the robot
        else:
            self.control_loop_timer_.cancel()   

def main(args=None):
    rclpy.init(args=args)
    MPC_controller_node = MPCCBFNode()
    rclpy.spin(MPC_controller_node)
    MPC_controller_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()