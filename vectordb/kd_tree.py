from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, TypeVar
from .db import DatabaseEntry, FloatType, Vector, VectorDatabase, VectorID
import numpy as np


@dataclass(frozen=True)
class _TreeProperties:
    dim: int
    num_splits_per_dimension: int
    max_leaf_size: int


def _get_split_dim(depth: int, dim: int) -> int:
    # hashing depth can be used to select the partition dimension pseudo-randomly
    split_dim = depth % dim
    return split_dim


T = TypeVar('T')
class _KDTree(Generic[T]):
    def __init__(self, lower: Vector, upper: Vector, depth: int, tree_properties: _TreeProperties):
        super().__init__()
        self._depth = depth
        self._tree_props = tree_properties
        self._lower = lower
        self._upper = upper
        
        self.is_leaf = True
        self.child_nodes: List[Optional[_KDTree]] = [None] * (self._tree_props.num_splits_per_dimension + 1)
        self.leaf_entries: Dict[VectorID, DatabaseEntry[T]] = {}
         
    def _create_child_tree(self, idx: int, split_dim: int, new_depth: int, current_lower: FloatType, current_upper: FloatType):
        num_splits = self._tree_props.num_splits_per_dimension
        num_partitions = num_splits + 1

        new_lower = (current_upper - current_lower) * (idx / num_partitions) + current_lower
        new_upper = (current_upper - current_lower) * ((idx + 1) / num_partitions) + current_lower

        new_lower_vec = self._lower.copy()
        new_lower_vec[split_dim] = new_lower

        new_upper_vec = self._upper.copy()
        new_upper_vec[split_dim] = new_upper

        child_tree = _KDTree(new_lower_vec, new_upper_vec, new_depth, self._tree_props)
        return child_tree
    
    def _get_child_index_impl(self, depth: int, position: int):
        new_depth = depth + 1
        split_dim = _get_split_dim(new_depth, self._tree_props.dim)
        lower: FloatType = self._lower[split_dim]
        upper: FloatType = self._upper[split_dim]

        split_value: FloatType = position[split_dim]
        normalized_split_value = (split_value - lower) / (upper - lower)
        num_splits = self._tree_props.num_splits_per_dimension
        idx = int(normalized_split_value * (num_splits + 1))
        idx = min(idx, num_splits) # (rare case that normalized_value is exactly 1 and smaller)
        return idx, split_dim, new_depth, lower, upper

    def _add_to_child(self, id: VectorID, entry: DatabaseEntry[T]):
        idx, split_dim, new_depth, lower, upper = self._get_child_index_impl(self._depth, entry.position)

        child_tree = self.child_nodes[idx]
        if child_tree is None:
            child_tree = self._create_child_tree(
                idx=idx,
                split_dim=split_dim,
                new_depth=new_depth,
                current_lower=lower, 
                current_upper=upper,
            )
            assert(np.all(entry.position >= child_tree._lower))
            assert(np.all(entry.position <= child_tree._upper))
            self.child_nodes[idx] = child_tree
        
        child_tree.insert(id, entry)

    def _get_child_index(self, depth: int, position: Vector):
        idx, *_ = self._get_child_index_impl(depth, position)
        return idx 

    def insert(self, id: VectorID, entry: DatabaseEntry[T]) -> VectorID:
        if self.is_leaf:
            self.leaf_entries[id] = entry

            if len(self.leaf_entries) > self._tree_props.max_leaf_size:
                for id, entry in self.leaf_entries.items():
                    self._add_to_child(id, entry)
                self.leaf_entries.clear()
                self.is_leaf = False
        else:
            self._add_to_child(id, entry)
    
    def __len__(self) -> int:
        if self.is_leaf:
            return len(self.leaf_entries)
        else:
            return sum([len(child) for child in self.child_nodes if child is not None])
    
    def get_depth(self):
        if self.is_leaf:
            return 0
        else:
            return 1 + max({child.get_depth() for child in self.child_nodes if child is not None})
    
    def get_temporary_path(self, position: Vector):
        path = [self]
        indices: List[int] = []
        while not (current_tree := path[-1]).is_leaf:
            child_idx = current_tree._get_child_index(current_tree._depth, position)
            path.append(current_tree.child_nodes[child_idx])
            indices.append(child_idx)
        return indices, path 
    
    def is_empty(self) -> bool:
        return len(self.leaf_entries) == 0 and all([node == None for node in self.child_nodes])


T = TypeVar('T')
class KDTreeDatabase(VectorDatabase[T]):
    def __init__(
            self,
            dim: int,
            lower_bound: np.ndarray[np.floating],
            upper_bound: np.ndarray[np.floating],
            num_splits_per_dimension: int,
            max_leaf_size: int,
    ):
        super().__init__()
        self._dim = dim 
        self._tree_props = _TreeProperties(dim, num_splits_per_dimension, max_leaf_size)

        lower_bound = lower_bound.astype(FloatType).copy()
        upper_bound = upper_bound.astype(FloatType).copy()
        self._tree = _KDTree(lower_bound, upper_bound, 0, self._tree_props)
        self._next_id = 0
        self._id_access: dict[VectorID, DatabaseEntry[T]] = {}

    def _create_unique_id(self) -> VectorID:
        new_id = self._next_id
        self._next_id += 1
        return new_id
    
    def dim(self):
        return self._dim
    
    def insert(self, position: Vector, metadata: T) -> VectorID:
        id = self._create_unique_id()
        entry = DatabaseEntry(position, metadata)
        self._tree.insert(id, entry)
        self._id_access[id] = entry
        return id 
    
    def get_entry(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        return self._id_access.get(id, None)
    
    def delete(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        entry = self._id_access.pop(id, None)
        if entry is None:
            return None 
        
        indices, path = self._tree.get_temporary_path(entry.position)
        leaf_node = path.pop()
        leaf_node.leaf_entries.pop(id)

        for tree, child_tree_idx in zip(reversed(path), reversed(indices)):
            child_tree = tree.child_nodes[child_tree_idx]
            if child_tree.is_empty():
                # -> can be removed from grid
                tree.child_nodes[child_tree_idx] = None 
            else:
                break
        
        # Make to leaf again if completely empty
        # Root node is the only node in the tree that is allowed to be completely empty
        if len(self) == 0:
            self._tree.is_leaf = True
        
        return entry 
    
    def __len__(self) -> int:
        return len(self._id_access)
    
    def _debug_compute_length_from_tree(self):
        return len(self._tree)
    
    def get_tree_depth(self) -> int:
        return self._tree.get_depth()