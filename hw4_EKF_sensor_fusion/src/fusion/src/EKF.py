import numpy as np
from math import cos, sin, atan2

class ExtendedKalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # Define what state to be estimate
        # Ex.
        #   only pose -> np.array([x, y, yaw])
        #   with velocity -> np.array([x, y, yaw, vx, vy, vyaw])
        #   etc...
        self.pose = np.array([x, y, yaw])   # only pose
        
        # Transition matrix
        self.A = np.identity(3) # jacobian matrix
        self.B = np.identity(3) # motion transition matrix
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.array([[1, 0, 0],[0, 1, 0], [0, 0, 1]])
        
        # State transition error
        self.R = np.identity(3) * 1
        
        # Measurement error
        self.Q = np.identity(3) * 1
        print("Initialize Kalman Filter")
    
    def predict(self, u):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear motion model for the control input
        # Calculate Jacobian matrix of the model as self.A
        # u = [del_x, del_y, del_yaw]

        self.B = np.array([[cos(self.pose[2]), -sin(self.pose[2]), 0],  
                           [sin(self.pose[2]), cos(self.pose[2]), 0],
                           [0, 0, 1]])  # setting the motion transition matrix

        self.A = np.array([[1, 0, -sin(self.pose[2])*u[0] - cos(self.pose[2])*u[1]],    
                           [0, 1,  cos(self.pose[2])*u[0] - sin(self.pose[2])*u[1]],
                           [0, 0, 1]])  # setting the jacobian matrix

        self.pose += self.B @ u # motion model
        self.S = self.A @ self.S @ self.A.T + self.R    # state
        
    def update(self, z):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear observation matrix for the measurement input
        # Calculate Jacobian matrix of the matrix as self.C
        # z = [x, y, yaw]

        # I choose the linear model to update the pose & state

        K = self.S @ self.C.T @ np.linalg.inv(self.C @ self.S @ self.C.T + self.Q)
        self.pose = self.pose + K @ (z - self.C @ self.pose)
        self.S = (np.identity(3) - K @ self.C) @ self.S

        return self.pose, self.S