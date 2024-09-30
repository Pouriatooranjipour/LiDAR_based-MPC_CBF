import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt  # Add this import for plotting
from math import pi
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped



class LidarClusteringNode(Node):
    def __init__(self):
        super().__init__('lidar_clustering_node')

        # Parameters
        self.declare_parameter('eps', 0.5)
        self.declare_parameter('min_samples', 15)

        # Get parameters
        self.eps = self.get_parameter('eps').value
        self.min_samples = self.get_parameter('min_samples').value

        # Subscribe to Lidar scan data
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan_filtered',
            self.scan_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publisher for ellipses
        self.ellipse_publisher = self.create_publisher(Float64MultiArray, '/lidar_ellipses', 10)
        # self.marker_publisher = self.create_publisher(MarkerArray, '/lidar_ellipses_plot', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
               # Transform listener

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)

        valid_indices = np.isfinite(ranges)
        ranges = ranges[valid_indices]
        angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        angles = angles[valid_indices]
        
        # Convert polar coordinates to Cartesian
        x_coords = ranges * np.cos(angles)
        y_coords = ranges * np.sin(angles)
        # List to store transformed points
        transformed_points = []

        for x, y in zip(x_coords, y_coords):
            # Create a PointStamped in the Lidar frame
            obstacle_point_lidar = PointStamped()
            obstacle_point_lidar.header.stamp = msg.header.stamp
            obstacle_point_lidar.header.frame_id = 'laser'
            obstacle_point_lidar.point.x = float(x)
            obstacle_point_lidar.point.y = float(y)
            obstacle_point_lidar.point.z = 0.0
            
            try:
                
                # Look up the transform from Lidar frame to Odom frame at the adjusted time
                transform = self.tf_buffer.lookup_transform(
                    'odom',
                    'laser',
                    rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1)
                )
                
                # Transform the point from Lidar frame to Odom frame
                obstacle_point_odom = do_transform_point(obstacle_point_lidar, transform)

                # Store transformed points
                transformed_points.append([obstacle_point_odom.point.x, obstacle_point_odom.point.y])

                # self.get_logger().info(f'Obstacle at (x: {obstacle_point_odom.point.x}, y: {obstacle_point_odom.point.y}) in odom frame')

            except Exception as e:
                self.get_logger().warn(f'Could not transform point: {e}')

        if transformed_points:
            points = np.array(transformed_points)
            # plt.figure()
            # plt.scatter(points[:, 0], points[:, 1])
            # plt.title('All Lidar Points')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.legend()
            # plt.show()
            dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = dbscan.fit_predict(points)

            unique_labels = set(labels)
            # marker_array = MarkerArray()
            ellipses = []

            for label in unique_labels:
                if label == -1:  # Ignore noise points
                    continue

                cluster_points = points[labels == label]
                # Plot each cluster
                # plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label}')

                # Fit ellipse
                if len(cluster_points) >= 5:  # OpenCV fitEllipse requires at least 5 points
                    ellipse = cv2.fitEllipse(cluster_points.astype(np.float32))
                    (x_c, y_c), (r_x, r_y), angle = ellipse

                    
                    ellipses.append([x_c, y_c, r_x, r_y])
                    # t = np.linspace(0, 2*pi, 100)
                    # plt.plot( x_c+r_x*np.cos(t) , y_c+r_y*np.sin(t) )
                    self.get_logger().info(f'Obstacle at (x: {x_c}, y: {y_c}, r_x:{r_x}, r_y:{r_y}) in odom frame')

                        # Add transformed ellipse properties to the list
                        # ellipses.append([transformed_center.point.x, transformed_center.point.y, r_x, r_y])

                    # except Exception as e:
                    #     self.get_logger().warn(f"Could not transform to odom frame: {e}")
                    #     continue

                    # # Add ellipse properties to the list
                    # ellipses.append([x_c, y_c, r_x, r_y])


                    # Create a marker for this ellipse
                    # marker = Marker()
                    # marker.header.frame_id = "odom"  # Adjust based on your frame
                    # marker.header.stamp = self.get_clock().now().to_msg()
                    # marker.type = Marker.SPHERE
                    # marker.action = Marker.ADD
                    # marker.pose.position.x = x_c
                    # marker.pose.position.y = y_c
                    # marker.pose.position.z = 0.0  # Assuming a 2D Lidar, z = 0

                    # Convert the angle from degrees to radians
                    # angle_rad = np.deg2rad(angle)
                    # marker.pose.orientation.z = np.sin(angle_rad / 2)
                    # marker.pose.orientation.w = np.cos(angle_rad / 2)

                    # marker.scale.x = r_x
                    # marker.scale.y = r_y
                    # marker.scale.z = 0.1  # Thin ellipse for 2D visualization

                    # marker.color.a = 0.5  # Transparency
                    # marker.color.r = 1.0
                    # marker.color.g = 0.0
                    # marker.color.b = 0.0

                    # marker.id = int(label)  # Use label as the ID

                    # marker_array.markers.append(marker)

            # Show the plot
            # plt.title('Lidar Cluster Points')
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.legend()
            # plt.show()       
            # Publish ellipse data
            ellipse_msg = Float64MultiArray()
            for ellipse in ellipses:
                ellipse_msg.data.extend(ellipse)

            self.ellipse_publisher.publish(ellipse_msg)
            # self.marker_publisher.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = LidarClusteringNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
