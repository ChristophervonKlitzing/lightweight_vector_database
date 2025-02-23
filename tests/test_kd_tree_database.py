from typing import List
import unittest

from vectordb.database import VectorID
from vectordb.kd_tree_database import KDTreeDatabase
import numpy as np


def _setup_test_db(dim = 4):
    db = KDTreeDatabase[str](
        dim=dim,
        lower_bound=np.zeros(dim),
        upper_bound=np.ones(dim),
        num_splits_per_dimension=2,
        max_leaf_size=5,
    )
    return db 

class TestKDTreeDatabase(unittest.TestCase):
    def test_insert_and_len(self):
        db = _setup_test_db()

        num_inserts = 100
        for i in range(num_inserts):
            db.insert(np.random.random(db.dim), f"data[{i}]")
        
        self.assertEqual(len(db), num_inserts)
        self.assertEqual(len(db), db._debug_compute_length_from_tree())

    def test_insert_delete_and_len(self):
        db = _setup_test_db()

        current_num_entries = 0
        max_num_entries = 100
        ids: List[VectorID] = []
        for i in range(max_num_entries):
            id = db.insert(np.random.random(db.dim), f"data[{i}]")
            ids.append(id)
            current_num_entries += 1

            self.assertEqual(len(db), current_num_entries)
            self.assertEqual(db._debug_compute_length_from_tree(), current_num_entries)

        for id in ids:
            db.delete(id)
            current_num_entries -= 1

            self.assertEqual(len(db), current_num_entries)
            self.assertEqual(db._debug_compute_length_from_tree(), current_num_entries)

        self.assertEqual(db.get_tree_depth(), 0)
        self.assertEqual(len(db), db._debug_compute_length_from_tree())

    def test_k_nearest_neighbors(self):
        db = _setup_test_db(2)
        num_inserts = 100
        for i in range(num_inserts):
            db.insert(np.random.rand(db.dim), f"data[{i}]")
        
        position = np.array([1 / 3] * db.dim)

        db.insert(position, "awd")
        neighbors = db.find_k_nearest_neighbors(position, k=10)
        entry, dist = neighbors[0]
        self.assertTrue(np.array_equal(entry.position, position))
        self.assertEqual(dist, 0.0)

        self.assertEqual(len(db), db._debug_compute_length_from_tree())
    
    def test_operations_on_empty(self):
        db = _setup_test_db()
        self.assertEqual(db.get_tree_depth(), 0)
        self.assertEqual(len(db), 0)
        neighbors = db.find_k_nearest_neighbors(np.zeros(db.dim), k=10)
        self.assertEqual(len(neighbors), 0)

        self.assertEqual(len(db), db._debug_compute_length_from_tree())
    
    def test_update_position(self):
        db = _setup_test_db()
        id = db.insert(np.zeros(db.dim), "awd")

        new_position = np.ones(db.dim)
        db.update_position(id, new_position)

        self.assertEqual(len(db), 1)
        neighbors = db.find_k_nearest_neighbors(np.zeros(db.dim), k=10)
        self.assertEqual(len(neighbors), 1)
        
        entry, dist = neighbors[0]
        self.assertGreater(dist, 0)
        self.assertTrue(np.array_equal(entry.position, new_position))

        self.assertEqual(len(db), db._debug_compute_length_from_tree())
    
    def test_iter(self):
        db = _setup_test_db()
        id_1 = db.insert(np.zeros(db.dim), "1")
        id_2 = db.insert(np.zeros(db.dim), "2")
        expected_ids = set([id_1, id_2])

        iterated_ids = set()
        for id, _ in db:
            iterated_ids.add(id)
        
        self.assertEqual(expected_ids, iterated_ids)
    
    def test_immutability(self):
        db = _setup_test_db()
        id = db.insert(np.zeros(db.dim), {"a": 0})

        def change_position():
            db.get_entry(id).position[0] = 0.
        
        self.assertRaises(ValueError, change_position)

        metadata = db.get_entry(id).metadata
        metadata["a"] = 1 # should only change a copy

        self.assertEqual(db.get_entry(id).metadata["a"], 0)

        

if __name__ == '__main__':
    unittest.main()
    