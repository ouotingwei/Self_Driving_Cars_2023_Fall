#!/usr/bin/env python3

import rospy
import os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from nav_msgs.msg import Odometry
from math import cos, sin, atan2, sqrt
from matplotlib import pyplot as plt

from EKF import ExtendedKalmanFilter

class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size = 10)
        self.EKF = None
        
        self.gt_list = [[], []]
        self.est_list = [[], []]
        self.initial = False

        # previous pose data from radar_odometry
        self.radar_x_previous = 0
        self.radar_y_previous = 0
        self.radar_yaw_previous = 0

        # previous gps data
        self.gps_pre_x = 0
        self.gps_pre_y = 0
        self.gps_pre_yaw = 0

    def shutdown(self):
        print("shuting down fusion.py")

    def predictPublish(self):
        
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        # change to the state x and state y from EKF
        predPose.pose.pose.position.x = self.EKF.pose[0]
        predPose.pose.pose.position.y = self.EKF.pose[1]
        
        # Change to the state yaw from EKF
        quaternion = quaternion_from_euler(0, 0, self.EKF.pose[2])
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]
        
        # Change to the covariance matrix of [x, y, yaw] from EKF
        predPose.pose.covariance = [self.EKF.S[0][0], self.EKF.S[0][1], 0, 0, 0, self.EKF.S[0][2],
                                    self.EKF.S[1][0], self.EKF.S[1][1], 0, 0, 0, self.EKF.S[1][2],
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    self.EKF.S[2][0], self.EKF.S[2][1], 0, 0, 0, self.EKF.S[2][2]]
    
        # rospy.loginfo("state Covariance:\n%s", predPose.pose.covariance)
                                    
        self.posePub.publish(predPose)

    def odometryCallback(self, data):
        odom_x = data.pose.pose.position.x
        odom_y = data.pose.pose.position.y

        odom_quaternion = [
            data.pose.pose.orientation.x, data.pose.pose.orientation.y,
            data.pose.pose.orientation.z, data.pose.pose.orientation.w
        ]
        _, _, odom_yaw = euler_from_quaternion(odom_quaternion)
        odom_covariance = np.array(data.pose.covariance).reshape(6, 6)
        
        # Design the control of EKF state from radar odometry data
        # The data is in global frame, you may need to find a way to convert it into local frame
        # Ex. 
        #     Find differnence between 2 odometry data 
        #         -> diff_x = ???
        #         -> diff_y = ???
        #         -> diff_yaw = ???
        #     Calculate transformation matrix between 2 odometry data
        #         -> transformation = last_odom_pose^-1 * current_odom_pose
        #     etc.

        delta_x = odom_x - self.radar_x_previous
        delta_y = odom_y - self.radar_y_previous
        delta_yaw = atan2(delta_y, delta_x)

        delta_yaw -= self.radar_yaw_previous

        del_dis = sqrt(pow(delta_x, 2) + pow(delta_y, 2))

        diff_x = del_dis * cos(delta_yaw)
        diff_y = del_dis * sin(delta_yaw)
        diff_yaw = odom_yaw - self.radar_yaw_previous
        
        control = [diff_x, diff_y, diff_yaw]

        self.radar_x_previous = odom_x
        self.radar_y_previous = odom_y
        self.radar_yaw_previous = odom_yaw
        
        if not self.initial:
            self.initial = True
            self.EKF = ExtendedKalmanFilter(odom_x, odom_y, odom_yaw)
        else:
            # Update error covriance
            self.EKF.R = np.array([
                [odom_covariance[0][0], odom_covariance[0][1], odom_covariance[0][5]],
                [odom_covariance[1][0], odom_covariance[1][1], odom_covariance[1][5]],
                [odom_covariance[5][0], odom_covariance[5][1], odom_covariance[5][5]]])*1000 # reduce a 6x6 matrix to a 3x3 matrix
            self.EKF.predict(u = control)

        self.predictPublish()
        
    def gpsCallback(self, data):
        gps_x = data.pose.pose.position.x
        gps_y = data.pose.pose.position.y
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)

        # Design the measurement of EKF state from GPS data
        # Ex. 
        #     Use GPS directly
        #     Find a approximate yaw
        #     etc.

        x_diff = gps_x - self.gps_pre_x
        y_diff = gps_y - self.gps_pre_y

        # Check if the difference is within 2
        if abs(x_diff) > 2 or abs(y_diff) > 2:
            gps_yaw = atan2(y_diff, x_diff)
            self.gps_pre_yaw = gps_yaw
            measurement = [gps_x, gps_y, gps_yaw]
        else:
            # If the difference is within 2, do not update the measurement
            measurement = [self.gps_pre_x, self.gps_pre_y, gps_yaw]

        self.gps_pre_x = gps_x
        self.gps_pre_y = gps_y

        if not self.initial:
            self.initial = ExtendedKalmanFilter(gps_x, gps_y)
            self.initial = True
        else:
            #Update error covriance
            self.EKF.Q = np.array([
                [gps_covariance[0][0], gps_covariance[0][1], gps_covariance[0][5]],
                [gps_covariance[1][0], gps_covariance[1][1], gps_covariance[1][5]],
                [gps_covariance[5][0], gps_covariance[5][1], gps_covariance[5][5]]])*1 # reduce a 6x6 matrix to a 3x3 matrix"
            
            # Update error covriance
            #self.EKF.Q = np.array([
            #    [gps_covariance[0][0], gps_covariance[0][1]],
            #    [gps_covariance[1][0], gps_covariance[1][1]]])*1 # reduce a 6x6 matrix to a 3x3 matrix
            self.EKF.update(z = measurement)

        self.predictPublish()
    
    def gtCallback(self, data):
        self.gt_list[0].append(data.pose.pose.position.x)
        self.gt_list[1].append(data.pose.pose.position.y)
        if self.EKF is not None:
            # Change to the state x and state y from EKF
            self.est_list[0].append(self.EKF.pose[0]) 
            self.est_list[1].append(self.EKF.pose[1])
        return

    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.plot(self.gt_list[0], self.gt_list[1], alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(self.est_list[0], self.est_list[1], alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        if not os.path.exists("./result"):
            print("not exist")
            os.mkdir("./result")
        plt.savefig("./result/result.png")
        plt.show()
        return
    
if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    fusion.plot_path()