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
        self.C = None
        
        # State transition error
        self.R = None
        
        # Measurement error
        self.Q = None

    def predict(self, u):
        raise NotImplementedError

    def update(self, z):
        raise NotImplementedError
        return self.x, self.S
