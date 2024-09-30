import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/pouria/Documents/Lidar_based_MPC_CBF_github/LiDAR_based-MPC_CBF/install/controller'
