import numpy as np
from math import cos, sin, atan2

class ExtendedKalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # Define what state to be estimate
        # Ex.
        #   only pose -> np.array([x, y, yaw])
        #   with velocity -> np.array([x, y, yaw, vx, vy, vyaw])
        #   etc...
        self.pose = np.array([x, y, yaw])
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3)
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.identity(3)
        
        # State transition error
        self.R = np.identity(3) * 1
        
        # Measurement error
        self.Q = np.identity(3) * 1
        print("Initialize Kalman Filter")
    
    def predict(self, u):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear motion model for the control input
        # Calculate Jacobian matrix of the model as self.A
        self.pose = self.A @ self.pose + self.B @ u
        self.S = self.A @ self.S @ self.A.T + self.R
        
    def update(self, z):
        # Base on the Kalman Filter design in Assignment 3
        # Implement a linear or nonlinear observation matrix for the measurement input
        # Calculate Jacobian matrix of the matrix as self.C
        K = self.S @ self.C.T @ np.linalg.inv(self.C @ self.S @ self.C.T + self.Q)
        self.pose = self.pose + K @ (z - self.C @ self.pose)
        self.S = (np.identity(3) - K @ self.C) @ self.S
        return self.pose, self.S