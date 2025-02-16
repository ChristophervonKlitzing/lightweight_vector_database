from dataclasses import dataclass
from typing import Callable, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar
import numpy as np

from .distance_metric import DistanceMetric, EuclideanDistance
from .database import DatabaseEntry, VectorDatabase
from .types import FloatType, Vector, VectorID



@dataclass(frozen=True)
class _TreeProperties:
    dim: int
    num_splits_per_dimension: int
    max_leaf_size: int


def _get_split_dim(depth: int, dim: int) -> int:
    # hashing depth can be used to select the partition dimension pseudo-randomly
    split_dim = depth % dim
    return split_dim

def _create_unit_vector(dim: int, position: int, value: float = 1.) -> Vector:
    vec = np.zeros(dim, dtype=FloatType)
    vec[position] = value
    return vec 


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
        self.leaf_vecs: Dict[VectorID, Vector] = {}
         
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

    def _add_to_child(self, id: VectorID, vec: Vector):
        idx, split_dim, new_depth, lower, upper = self._get_child_index_impl(self._depth, vec)

        child_tree = self.child_nodes[idx]
        if child_tree is None:
            child_tree = self._create_child_tree(
                idx=idx,
                split_dim=split_dim,
                new_depth=new_depth,
                current_lower=lower, 
                current_upper=upper,
            )
            assert(np.all(vec >= child_tree._lower))
            assert(np.all(vec <= child_tree._upper))
            self.child_nodes[idx] = child_tree
        
        child_tree.insert(id, vec)

    def _get_child_index(self, depth: int, position: Vector):
        idx, *_ = self._get_child_index_impl(depth, position)
        return idx 

    def insert(self, id: VectorID, vec: Vector) -> VectorID:
        if self.is_leaf:
            self.leaf_vecs[id] = vec

            if len(self.leaf_vecs) > self._tree_props.max_leaf_size:
                for id, vec in self.leaf_vecs.items():
                    self._add_to_child(id, vec)
                self.leaf_vecs.clear()
                self.is_leaf = False
        else:
            self._add_to_child(id, vec)
    
    def __len__(self) -> int:
        if self.is_leaf:
            return len(self.leaf_vecs)
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
    
    def delete(self, id: VectorID, position: Vector):
        indices, path = self.get_temporary_path(position)
        leaf_node = path.pop()
        leaf_node.leaf_vecs.pop(id)

        for tree, child_tree_idx in zip(reversed(path), reversed(indices)):
            child_tree = tree.child_nodes[child_tree_idx]
            if child_tree.is_empty():
                # -> can be removed from grid
                tree.child_nodes[child_tree_idx] = None 
            else:
                break
        
        # Make to leaf again if completely empty
        # Root node is the only node in the tree that is allowed to be completely empty
        is_leaf = all([c is None for c in self.child_nodes])
        if is_leaf:
            self.is_leaf = True
    
    def is_empty(self) -> bool:
        return len(self.leaf_vecs) == 0 and all([node == None for node in self.child_nodes])

    def find_k_nearest_neighbors(
            self, 
            position: Vector, 
            k: int, 
            filter: Optional[Callable[[VectorID], bool]] = None,
            distance_metric: Optional[DistanceMetric] = None,
    ):
        dim = self._tree_props.dim

        Q = TypeVar('Q')
        def _get_max_k(l: list[Q], k: int):
            return l if len(l) <= k else l[:k]
        
        sort_key = lambda x: x[-1]

        def distance_to_partition(tree: _KDTree, child_idx: int, split_dim: int, partition_idx: int) -> np.floating:
            # Compute the distance from the position to the partition defined by partition_idx
            # Each partition has two plane-boundaries but depending on whether the partition_idx is smaller or greater
            # than child_idx, the one or the other boundary has to be selected for distance computation.
            if child_idx == partition_idx:
                # same partition -> distance zero
                return 0.
            elif partition_idx < child_idx:
                plane_coordinate = tree.child_nodes[partition_idx]._upper[split_dim]
            else: # >
                plane_coordinate = tree.child_nodes[partition_idx]._lower[split_dim]
            
            # Build normal- and support-vector and compute position to plane distance
            normal = _create_unit_vector(dim, split_dim)
            support = _create_unit_vector(dim, split_dim, plane_coordinate) # x_0
            # 0 = n^T (x - x_0) = x_{split_index} - plane_coordinate <=> x_{split_index} = plane_coordinate
            squared_dist = distance_metric.squared_point_2_plane_distance(position, normal, support)
            return squared_dist

        def _find_k_nearest_neighbors(tree: _KDTree, neighbors: list[tuple[VectorID, np.floating]]):
            if tree.is_leaf:
                squared_dist = distance_metric.squared_point_2_point_distance
                entries_with_distances = [
                    (id, squared_dist(position, vec))
                    for id, vec in tree.leaf_vecs.items()
                    if filter(id)
                ]
                entries_with_distances.sort(key=sort_key)

                entries_with_distances = _get_max_k(entries_with_distances, k)
                new_neighbors = sorted(entries_with_distances + neighbors, key=sort_key)
                return _get_max_k(new_neighbors, k)
            else:
                child_idx, split_dim, *_ = tree._get_child_index_impl(tree._depth, position)
                children = [
                    (child_node, distance_to_partition(tree, child_idx, split_dim, i))
                    for i, child_node in enumerate(tree.child_nodes) 
                    if child_node is not None
                ]
                children.sort(key=sort_key)
                children = _get_max_k(children, k) # each child contains at least one element -> max k children
                current_neighbors = neighbors # assert always sorted in ascending order
                for child, squared_dist in children: # iterate in ascending order
                    if  len(current_neighbors) == 0 or squared_dist <= current_neighbors[-1][-1] or len(current_neighbors) < k:
                        current_neighbors = _find_k_nearest_neighbors(child, current_neighbors)
                    else:
                        break # box_dist is only increasing and the found current neighbors have all a smaller distance than the boxes
                return current_neighbors
        
        return _find_k_nearest_neighbors(self, [])
        

T = TypeVar('T')
class KDTreeDatabase(VectorDatabase[T]):
    """
    KD-tree like partitioning except that the interval (hyper-box) is devided into 
    equally sized sub-spaces (not along the median) and more than two partitions are made per dimension.

    This class requires maximally (max_leaf_size - 1) many points to be at the exact
    same location. Furthermore, the tree will only be balanced if the data is equally distributed.
    In turn, it provides k nearest neighbor querying with a customizable distance metric.
    """
    def __init__(
            self,
            dim: int,
            lower_bound: np.ndarray[np.floating],
            upper_bound: np.ndarray[np.floating],
            num_splits_per_dimension: int,
            max_leaf_size: int,
    ):
        super().__init__()
        self._default_dist_metric = EuclideanDistance()
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
    
    @property
    def dim(self):
        return self._dim
    
    def insert(self, position: Vector, metadata: T) -> VectorID:
        id = self._create_unique_id()
        self._tree.insert(id, position)
        self._id_access[id] = DatabaseEntry(position, metadata)
        return id 
    
    def get_entry(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        return self._id_access.get(id, None)
    
    def delete(self, id: VectorID) -> Optional[DatabaseEntry[T]]:
        entry = self._id_access.pop(id, None)
        if entry is None:
            return None 
        self._tree.delete(id, entry.position)
        return entry 

    def find_k_nearest_neighbors(
        self, 
        position: Vector, 
        k: int, 
        filter: Optional[Callable[[T], bool]] = None,
        distance_metric: Optional[DistanceMetric] = None,
    ) -> Sequence[Tuple[DatabaseEntry[T], np.floating]]:
        # The tree doesn't contain metadata

        if filter is None:
            filter_with_id = lambda id: True 
        else:
            filter_with_id = lambda id: filter(self._id_access[id].metadata)
        
        if distance_metric is None:
            distance_metric = self._default_dist_metric
        
        ids_and_squared_dists = self._tree.find_k_nearest_neighbors(position, k, filter_with_id, distance_metric)
        neighbors = [(self._id_access[id], squared_dist) for id, squared_dist in ids_and_squared_dists]
        return neighbors
    
    def __len__(self) -> int:
        return len(self._id_access)
    
    def update_position(self, id: VectorID, new_position: Vector):
        entry = self._id_access.get(id)
        self._id_access[id] = DatabaseEntry(new_position, entry.metadata)
        self._tree.delete(id, entry.position)
        self._tree.insert(id, new_position)
    
    def _debug_compute_length_from_tree(self):
        return len(self._tree)
    
    def get_tree_depth(self) -> int:
        return self._tree.get_depth()