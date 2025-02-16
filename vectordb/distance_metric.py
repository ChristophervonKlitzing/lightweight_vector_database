from abc import ABC, abstractmethod
import numpy as np
from .types import Vector


class DistanceMetric(ABC):
    @abstractmethod
    def squared_point_2_point_distance(self, p1: Vector, p2: Vector) -> np.floating:
        raise NotImplementedError
    
    @abstractmethod
    def squared_point_2_plane_distance(self, p: Vector, n: Vector, x_0) -> np.floating:
        """
        Computes the squared distance of the given point p to the plane defined by 
        $n^T^ * (x - x_0) = 0$ where x is any point on the plane.

        :params:
            - p: point from which to compute the distance

            - n: normal vector of the plane

            - x_0: A point in the plane that defines its offset from the origin
        """
        raise NotImplementedError


class EuclideanDistance(DistanceMetric):
    def squared_point_2_point_distance(self, p1: Vector, p2: Vector) -> np.floating:
        vec = p2 - p1
        return np.dot(vec, vec)
    
    def squared_point_2_plane_distance(self, p: Vector, n: Vector, x_0) -> np.floating:
        return np.dot(n, p - x_0)**2 / np.dot(n, n)


class MahalanobisDistance(DistanceMetric):
    def __init__(self, covariance: np.ndarray[np.floating]):
        super().__init__()
        self.S = covariance

        if len(covariance.shape) == 1:
            self.S_inv = 1 / covariance # diagonal
        else:
            self.S_inv = None # solved through np.linalg.sove
    
    def squared_point_2_point_distance(self, p1: Vector, p2: Vector) -> np.floating:
        vec = p2 - p1
        if self.S_inv is None:
            z = np.linalg.solve(self.S, vec)
            return np.dot(vec, z)
        else:
            return np.dot(vec, vec * self.S_inv)
    
    def squared_point_2_plane_distance(self, p: Vector, n: Vector, x_0) -> np.floating:
        a = np.dot(n, p - x_0)**2
        if self.S_inv is None:
            n_vec = np.expand_dims(n, 1)
            b = np.squeeze(n_vec.T @ self.S @ n_vec)
        else:
            b = np.dot(n, self.S * n)
        return a / b 
