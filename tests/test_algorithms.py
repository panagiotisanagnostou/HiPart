import numpy as np
import pickle
import unittest


from HiPart.clustering  import dePDDP
from HiPart.clustering  import iPDDP
from HiPart.clustering  import kM_PDDP
from HiPart.clustering  import PDDP
from HiPart.clustering  import bicecting_kmeans
from unittest import TestCase


class Test_Validate_Execution(TestCase):
    ran_state = 1256
    
    def setUp(self):
        with open("assets/test_data.dump", "rb") as inf:
            self.test_data = pickle.load(inf)
        
    def test_dePDDP_return_type(self):
        new_obj = dePDDP(
            max_clusters_number=3,
        ).fit(self.test_data["data"])
        self.assertIsInstance(new_obj, dePDDP)
    
    def test_iPDDP_return_type(self):
        new_obj = iPDDP(
            max_clusters_number=3,
        ).fit(self.test_data["data"])
        self.assertIsInstance(new_obj, iPDDP)
    
    def test_kM_PDDP_return_type(self):
        new_obj = kM_PDDP(
            max_clusters_number=3,
        ).fit(self.test_data["data"])
        self.assertIsInstance(new_obj, kM_PDDP)
    
    def test_PDDP_return_type(self):
        new_obj = PDDP(
            max_clusters_number=3,
        ).fit(self.test_data["data"])
        self.assertIsInstance(new_obj, PDDP)
    
    def test_bicecting_kmeans_return_type(self):
        new_obj = bicecting_kmeans(
            max_clusters_number=3,
        ).fit(self.test_data["data"])
        self.assertIsInstance(new_obj, bicecting_kmeans)
    
    def test_bicecting_kmeans_results(self):
        matrix_control = self.test_data["bicecting_kmeans"]
     
        matrix_test = bicecting_kmeans(
            max_clusters_number=3,
            random_seed=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)

    def test_dePDDP_pca_results(self):
        matrix_control = self.test_data["dePDDP_pca"]
        
        matrix_test = dePDDP(
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_iPDDP_pca_results(self):
        matrix_control = self.test_data["iPDDP_pca"]
     
        matrix_test = iPDDP(
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_kM_PDDP_pca_results(self):
        matrix_control = self.test_data["kM_PDDP_pca"]
     
        matrix_test = kM_PDDP(
            max_clusters_number=3,
            random_seed=self.ran_state,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_PDDP_pca_results(self):
        matrix_control = self.test_data["PDDP_pca"]
     
        matrix_test = PDDP(
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
        
    def test_dePDDP_ica_results(self):
        matrix_control = self.test_data["dePDDP_ica"]
        
        matrix_test = dePDDP(
            decomposition_method="ica", 
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_iPDDP_ica_results(self):
        matrix_control = self.test_data["iPDDP_ica"]
     
        matrix_test = iPDDP(
            decomposition_method="ica",
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_kM_PDDP_ica_results(self):
        matrix_control = self.test_data["kM_PDDP_ica"]
     
        matrix_test = kM_PDDP(
            decomposition_method="ica",
            max_clusters_number=3,
            random_seed=self.ran_state,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_PDDP_ica_results(self):
        matrix_control = self.test_data["PDDP_ica"]
     
        matrix_test = PDDP(
            decomposition_method="ica",
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
        
    def test_dePDDP_kpca_results(self):
        matrix_control = self.test_data["dePDDP_kpca"]
        
        matrix_test = dePDDP(
            decomposition_method="kpca",
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_iPDDP_kpca_results(self):
        matrix_control = self.test_data["iPDDP_kpca"]
     
        matrix_test = iPDDP(
            decomposition_method="kpca",
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_kM_PDDP_kpca_results(self):
        matrix_control = self.test_data["kM_PDDP_kpca"]
     
        matrix_test = kM_PDDP(
            decomposition_method="kpca",
            max_clusters_number=3,
            random_seed=self.ran_state,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)
    
    def test_PDDP_kpca_results(self):
        matrix_control = self.test_data["PDDP_kpca"]
     
        matrix_test = PDDP(
            decomposition_method="kpca",
            max_clusters_number=3,
            random_state=self.ran_state,
        ).fit(self.test_data["data"]).output_matrix
        self.assertEqual(np.sum(matrix_test == matrix_control), 300)


if __name__ == '__main__':
    unittest.main()