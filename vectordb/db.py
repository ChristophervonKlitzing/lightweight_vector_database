from abc import abstractmethod, ABC
from dataclasses import dataclass 
import numpy as np 
from typing import Callable, Generic, Iterable, Optional, TypeVar


VectorID = int 
FloatType = np.float32
Vector = np.ndarray[FloatType]


T = TypeVar('T')
@dataclass(frozen=True)
class DatabaseEntry(Generic[T]):
    position: Vector
    metadata: T 


T = TypeVar('T')
class VectorDatabase(ABC, Generic[T]):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def insert(self, position: Vector, metadata: T) -> VectorID:
        raise NotImplementedError
    
    """
    @abstractmethod
    def find_k_nearest_neighbors(
        self, 
        position: Vector, 
        k: int, 
        filter: Optional[Callable[[T], bool]] = None
    ) -> Iterable[DatabaseEntry[T]]:
        raise NotImplementedError

    @abstractmethod
    def update_position(self, id: VectorID, new_position: Vector):
        raise NotImplementedError
    
    """

    @abstractmethod
    def delete(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        raise NotImplementedError

    @abstractmethod
    def get_entry(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        raise NotImplementedError
    

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError