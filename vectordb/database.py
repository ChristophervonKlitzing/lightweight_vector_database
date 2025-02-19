from abc import abstractmethod, ABC
from dataclasses import dataclass 
from typing import Callable, Generic, Optional, Sequence, Tuple, TypeVar
import numpy as np

from .distance_metric import DistanceMetric
from .types import Vector, VectorID


T = TypeVar('T')
@dataclass(frozen=True)
class DatabaseEntry(Generic[T]):
    position: Vector
    metadata: T 


T = TypeVar('T')
class VectorDatabase(ABC, Generic[T]):
    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def insert(self, position: Vector, metadata: T) -> VectorID:
        raise NotImplementedError
    
    @abstractmethod
    def find_k_nearest_neighbors(
        self, 
        position: Vector, 
        k: int, 
        filter: Optional[Callable[[T], bool]] = None,
        distance_metric: Optional[DistanceMetric] = None,
    ) -> Sequence[Tuple[DatabaseEntry[T], np.floating]]:
        """
        Computes the k nearest neighbors to the given position that 
        satisfy the filter (filter returns True) using the defined distance metric.
        The default distance metric (if none is specified) is the default distance
        metric is used.

        Returns:
            The k nearest neighbors and their distances to the given position.
        """
        raise NotImplementedError

    @abstractmethod
    def update_position(self, id: VectorID, new_position: Vector):
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        raise NotImplementedError
    
    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError
    