"""
@author: OU,TING-WEI @ M.S. in Robotics 
date : 10-24-2023
Self-Driving-Cars HW3 ( NYCU FALL-2023 )
"""
import numpy as np

class KalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.state = np.array([x, y, yaw])
        
        # Transition matrix
        self.A = np.identity(3)  # Identity matrix for state transition
        self.B = np.identity(3)  # Identity matrix for control input
        
        # State covariance matrix
        self.S = np.identity(3) * 1  # Initialize state covariance matrix
        
        # Observation matrix
        self.C = np.array([[1, 0, 0],
                           [0, 1, 0]])  # Maps the state to the measurement space
        
        # State transition error
        self.R = np.identity(3) * 1  # Process noise covariance matrix
        
        # Measurement error
        self.Q = np.identity(2) * 3  # Measurement noise covariance matrix

    def predict(self, u):
        # Prediction step
        self.state = self.A @ self.state + self.B @ u
        self.S = self.A @ self.S @ self.A.T + self.R  # Update state covariance

    def update(self, z):
        # Update step
        K = self.S @ self.C.T @ np.linalg.inv(self.C @ self.S @ self.C.T + self.Q)
        self.state = self.state + K @ (z - self.C @ self.state)
        self.S = (np.identity(3) - K @ self.C) @ self.S  # Update state covariance
        return self.state, self.S  # Return the updated state and covariance