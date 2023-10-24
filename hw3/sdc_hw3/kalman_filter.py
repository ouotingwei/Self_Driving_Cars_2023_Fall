import numpy as np

class KalmanFilter:
    def __init__(self, x=0, y=0, yaw=0):
        # State [x, y, yaw]
        self.state = np.array([x, y, yaw])
        
        # Transition matrix
        self.A = np.identity(3)
        self.B = np.identity(3) 
        
        # State covariance matrix
        self.S = np.identity(3) * 1
        
        # Observation matrix
        self.C = np.array([[1, 0, 0],
                           [0, 1, 0]])
        
        # State transition error
        self.R = np.identity(3) 
        
        # Measurement error
        self.Q = np.identity(2) * 0.6

    def predict(self, u):
        self.state = self.A @ self.state + self.B @ u
        self.S = self.A @ self.S @ self.A.T + self.R

    def update(self, z):
        K = self.S @ self.C.T @ np.linalg.inv(self.C @ self.S @ self.C.T + self.Q)
        self.state = self.state + K @ (z - self.C @ self.state)
        self.S = (np.identity(3) - K @ self.C) @ self.S
        return self.state, self.S
