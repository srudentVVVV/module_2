"""Unit-тести для модуля data_loader.py — завантаження даних."""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_loader import DataLoader


class TestDataLoaderNormal(unittest.TestCase):
    """Тести нормальних умов для DataLoader."""

    def setUp(self):
        self.loader = DataLoader()

    def test_load_iris_returns_dict(self):
        """TC-D01: load_iris повертає словник з очікуваними ключами."""
        result = self.loader.load_iris()
        expected_keys = {"data", "labels", "feature_names", "target_names",
                         "n_samples", "n_features", "n_classes"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_iris_sample_count(self):
        """TC-D02: Iris містить 150 зразків."""
        result = self.loader.load_iris()
        self.assertEqual(result["n_samples"], 150)

    def test_iris_feature_count(self):
        """TC-D03: Iris містить 4 ознаки."""
        result = self.loader.load_iris()
        self.assertEqual(result["n_features"], 4)

    def test_iris_class_count(self):
        """TC-D04: Iris містить 3 класи."""
        result = self.loader.load_iris()
        self.assertEqual(result["n_classes"], 3)

    def test_data_shape(self):
        """TC-D05: Дані мають правильну форму (150, 4)."""
        result = self.loader.load_iris()
        self.assertEqual(result["data"].shape, (150, 4))

    def test_labels_shape(self):
        """TC-D06: Мітки мають правильну форму (150,)."""
        result = self.loader.load_iris()
        self.assertEqual(result["labels"].shape, (150,))


class TestDataLoaderNormalization(unittest.TestCase):
    """Тести нормалізації даних."""

    def setUp(self):
        self.loader = DataLoader()
        self.loader.load_iris()

    def test_normalized_range_min(self):
        """TC-D07: Нормалізовані дані >= 0."""
        self.assertTrue(np.all(self.loader.normalized_data >= 0))

    def test_normalized_range_max(self):
        """TC-D08: Нормалізовані дані <= 1."""
        self.assertTrue(np.all(self.loader.normalized_data <= 1))

    def test_normalized_contains_zero(self):
        """TC-D09: Граничне — нормалізовані дані містять 0 (мінімум ознаки)."""
        self.assertTrue(np.any(self.loader.normalized_data == 0))

    def test_normalized_contains_one(self):
        """TC-D10: Граничне — нормалізовані дані містять 1 (максимум ознаки)."""
        self.assertTrue(np.any(self.loader.normalized_data == 1))


class TestDataLoaderSummary(unittest.TestCase):
    """Тести текстового підсумку."""

    def test_summary_before_load(self):
        """TC-D11: Підсумок до завантаження — повідомлення про відсутність даних."""
        loader = DataLoader()
        summary = loader.get_summary()
        self.assertEqual(summary, "Дані не завантажені.")

    def test_summary_after_load(self):
        """TC-D12: Підсумок після завантаження містить назву датасету."""
        loader = DataLoader()
        loader.load_iris()
        summary = loader.get_summary()
        self.assertIn("Iris", summary)
        self.assertIn("150", summary)
        self.assertIn("4", summary)

    def test_summary_contains_class_names(self):
        """TC-D13: Підсумок містить назви класів."""
        loader = DataLoader()
        loader.load_iris()
        summary = loader.get_summary()
        self.assertIn("setosa", summary)
        self.assertIn("versicolor", summary)
        self.assertIn("virginica", summary)


if __name__ == "__main__":
    unittest.main()
