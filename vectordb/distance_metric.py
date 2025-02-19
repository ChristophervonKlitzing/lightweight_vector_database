from abc import ABC, abstractmethod
import numpy as np
from .types import Vector


class DistanceMetric(ABC):
    def point2plane_blackbox(self, p: Vector, n: Vector, x_0: Vector) -> np.floating:
        import scipy
        import scipy.optimize

        def f(x: Vector):
            return self.point2point(p, x)
        
        def constraint(x: Vector):
            return np.dot(n, x - x_0)
    
        res = scipy.optimize.minimize(f, x_0, constraints=[{"type": "eq", "fun": constraint}], method='trust-constr')
        distance: np.floating = res.fun
        return distance

    @abstractmethod
    def point2point(self, p1: Vector, p2: Vector) -> np.floating:
        raise NotImplementedError
    
    def point2plane(self, p: Vector, n: Vector, x_0: Vector) -> np.floating:
        """
        Computes the minimum of the following problem where n^T (x - x_0) = 0
        defines the plane and p is the point from which the distance to the plane should
        be computed.

        P: dist(p, x) s.t. n^T (x - x_0) = 0 (convex for any norm ||.||).
        dist(a, b) is computed from ||.|| for example dist(a, b)=||a-b|| or dist(a, b)=||a-b||^2.

        This function has a default implementation by using
        constraint optimization (e.g. from scipy).
        The underlying optimization problem is convex which implies that a globally
        minimal point and coresponding distance can be found.

        However, the default implementation relies on an optimization algorithm
        and treats the norm as a blackbox function. For better performance, this 
        function should be implemented manually using a closed form solution to the problem P.
        Furthermore, this default implementation can result in numerical inaccuracies
        if for example the distance is not differentiable everywhere (e.g. 1-norm, infinity-norm).
        These inaccuracies can be quite large which make the 
        output of algorithms like k-nearest neighbors wrong!

        :params:
            - p: point from which to compute the distance

            - n: normal vector of the plane

            - x_0: A point in the plane that defines its offset from the origin
        """
        return self.point2plane_blackbox(p, n, x_0)


class EuclideanDistance(DistanceMetric):
    def point2point(self, p1: Vector, p2: Vector) -> np.floating:
        vec = p2 - p1
        return np.dot(vec, vec)
    
    def point2plane(self, p: Vector, n: Vector, x_0) -> np.floating:
        return np.dot(n, p - x_0)**2 / np.dot(n, n)


class MahalanobisDistance(DistanceMetric):
    def __init__(self, covariance: np.ndarray[np.floating]):
        super().__init__()
        self.S = covariance

        if len(covariance.shape) == 1:
            self.S_inv = 1 / covariance # diagonal
        else:
            self.S_inv = None # solved through np.linalg.sove
    
    def point2point(self, p1: Vector, p2: Vector) -> np.floating:
        vec = p2 - p1
        if self.S_inv is None:
            z = np.linalg.solve(self.S, vec)
            return np.dot(vec, z)
        else:
            return np.dot(vec, vec * self.S_inv)
    
    def point2plane(self, p: Vector, n: Vector, x_0: Vector) -> np.floating:
        a = np.dot(n, p - x_0)**2
        if self.S_inv is None:
            n_vec = np.expand_dims(n, 1)
            b = np.squeeze(n_vec.T @ self.S @ n_vec)
        else:
            b = np.dot(n, self.S * n)
        distance = a / b 
        return distance


class InfinityNormDistance(DistanceMetric):
    """
    Other names: uniform norm, supremum norm, chebyshev norm (or max norm if the supremum is also the maximum).
    This implementation assumes the maximum is existing.
    The distance with this norm is defined as dist_{\inf}(p, q) = || p - q ||_{\inf}.

    || p - q ||_{\inf} = max {|p_1-q_1|, ..., |p_n-q_n|}
    """
    def point2point(self, p1: Vector, p2: Vector) -> np.floating:
        return np.max(np.abs(p1 - p2), axis=-1)


class OneNormDistance(DistanceMetric):
    def point2point(self, p1: Vector, p2: Vector) -> np.floating:
        return np.sum(np.abs(p1 - p2))
    
    def point2plane(self, p: Vector, n: Vector, x_0: Vector) -> np.floating:
        return (np.abs(np.dot(n, p - x_0)) / np.abs(n).max())
