from abc import ABC, abstractmethod
import numpy as np
import numpy.linalg
import numpy.random
import random as rand

# Inherit from this class or DIE
class Model(ABC):
    @abstractmethod
    def estimate_model_params(self, data_idx):
        return None
    
    @abstractmethod    
    def data_fit_error(self, model_params, data_idx):
        return None
    
    @abstractmethod
    def n_data(self):
        return None
    
class RansacTest(Model):
    def __init__(self, size=100, mu=0, sigma=1, num_outliers=30):
        self.size = size
        self.x = np.arange(self.size)
        self.y_true = self.x
        self.y = self.y_true + np.random.normal(mu, sigma, self.size)
        outliers = np.array(rand.sample(range(self.size), k=num_outliers))
        self.y[outliers] = np.random.uniform(0, size, num_outliers)

        
    def estimate_model_params(self, data_idx):
        A = np.block([[-self.x[data_idx]], 
                      [-np.ones(len(data_idx))],
                      [self.y[data_idx]]]).T
        V = np.linalg.svd(A)[2].T
        p = V[:,-1]
        p = p[0:2] / p[2]
        return p
    
    def validate_estimation(self):
        A = np.block([[-self.x], 
                      [-np.ones(self.size)],
                      [self.y_true]]).T
        V = np.linalg.svd(A)[2].T
        p = V[:,-1]
        p = p[0:2] / p[2]
        return p
    
    def data_fit_error(self, model_params, data_idx):
        a, b = model_params
        y_est = a*self.x[data_idx] + b
        y_true = self.y[data_idx]
        return np.abs(y_est - y_true)
    
    def n_data(self):
        return self.size
        

class ProjectionDLT(Model):
    def __init__(self, x_img, x_world):
        self.x_world = x_world
        self.x_img = x_img
        n_data1, self.n_img = x_img.shape
        n_data2, self.n_world = x_world.shape
        assert n_data1 == n_data2 # Number of data points must be equal
        self.n_data = n_data1
        self.P = np.empty((self.n_img, self.n_world))
        
    def estimate_model_params(self, data_idx):
        # Extract required column vectors
        X, Y, Z, W = x_world[0, data_idx].reshape((-1,1)), x_world[1, data_idx].reshape((-1,1)), x_world[2, data_idx].reshape((-1,1)), x_world[3, data_idx].reshape((-1,1))
        x, y, w = x_img[0, data_idx].reshape((-1,1)), x_img[1, data_idx].reshape((-1,1)), x_img[2, data_idx].reshape((-1,1))
        
        # Build A.
        n = len(data_idx)
        rows = 2*n
        cols = np.prod(self.P.shape) # Should be 12
        assert cols == 12
        A = np.zeros((rows, cols))
        _ZEROS = np.zeros((n, 1))
        A[0:2:,:] = np.block([X, Y, Z, W, _ZEROS, _ZEROS, _ZEROS, _ZEROS, -X*x, -Y*x, -Z*x, -W*x])
        A[1:2:,:] = np.block([_ZEROS, _ZEROS, _ZEROS, _ZEROS, X, Y, Z, W, -X*y, -Y*y, -Z*y, -W*y])
        
        # DLT and return estimated P
        V = np.linalg.svd(A)[2].T
        p = V[:,-1]
        self.P = p.reshape(self.P.shape)
        return self.P
    
    def data_fit_error(self, model_params, data_idx):
        P = model_params
        # Get estimated projection and actual projections
        x_est = P @ self.x_world[data_idx, :].T
        x_img = self.x_img[data_idx, :].T
        # Calculate reprojection error as quantitative measure of how well the model fits
        err = np.linalg.norm(x_est - x_img, axis=1)
        return err
    
    def n_data(self):
        return self.n_data