from typing import List
import unittest

from vectordb.db import VectorID
from vectordb.kd_tree import KDTreeDatabase
import numpy as np


def _setup_test_db():
    dim = 4
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
            db.insert(np.random.random(db.dim()), f"data[{i}]")
        
        self.assertEqual(len(db), num_inserts)

    def test_insert_delete_and_len(self):
        db = _setup_test_db()

        current_num_entries = 0
        max_num_entries = 100
        ids: List[VectorID] = []
        for i in range(max_num_entries):
            id = db.insert(np.random.random(db.dim()), f"data[{i}]")
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



if __name__ == '__main__':
    unittest.main()
    