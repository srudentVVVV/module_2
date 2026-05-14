"""Unit-тести для модуля kohonen.py — мережа Кохонена (SOM)."""

import unittest
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kohonen import KohonenSOM
from src.logger_setup import setup_logger

setup_logger("output/test.log")


class TestKohonenInit(unittest.TestCase):
    """Тести ініціалізації мережі Кохонена."""

    def test_normal_init(self):
        """TC-K01: Нормальна ініціалізація з коректними параметрами."""
        som = KohonenSOM(rows=5, cols=5, input_dim=4)
        self.assertEqual(som.rows, 5)
        self.assertEqual(som.cols, 5)
        self.assertEqual(som.input_dim, 4)
        self.assertEqual(som.weights.shape, (5, 5, 4))
        self.assertFalse(som.is_trained)

    def test_min_size_map(self):
        """TC-K02: Граничне значення — карта 1x1."""
        som = KohonenSOM(rows=1, cols=1, input_dim=2)
        self.assertEqual(som.weights.shape, (1, 1, 2))

    def test_large_map(self):
        """TC-K03: Граничне значення — велика карта 50x50."""
        som = KohonenSOM(rows=50, cols=50, input_dim=4)
        self.assertEqual(som.weights.shape, (50, 50, 4))

    def test_weights_in_range(self):
        """TC-K04: Ваги ініціалізовані в діапазоні [0, 1]."""
        som = KohonenSOM(rows=10, cols=10, input_dim=4)
        self.assertTrue(np.all(som.weights >= 0))
        self.assertTrue(np.all(som.weights <= 1))

    def test_reproducibility_with_seed(self):
        """TC-K05: Відтворюваність результатів з однаковим seed."""
        som1 = KohonenSOM(rows=5, cols=5, input_dim=4, random_seed=42)
        som2 = KohonenSOM(rows=5, cols=5, input_dim=4, random_seed=42)
        np.testing.assert_array_equal(som1.weights, som2.weights)

    def test_different_seeds_differ(self):
        """TC-K06: Різні seed дають різні ваги."""
        som1 = KohonenSOM(rows=5, cols=5, input_dim=4, random_seed=42)
        som2 = KohonenSOM(rows=5, cols=5, input_dim=4, random_seed=99)
        self.assertFalse(np.array_equal(som1.weights, som2.weights))


class TestKohonenBMU(unittest.TestCase):
    """Тести пошуку BMU."""

    def setUp(self):
        self.som = KohonenSOM(rows=3, cols=3, input_dim=2, random_seed=42)

    def test_bmu_returns_tuple(self):
        """TC-K07: BMU повертає кортеж (row, col)."""
        sample = np.array([0.5, 0.5])
        bmu = self.som._find_bmu(sample)
        self.assertIsInstance(bmu, tuple)
        self.assertEqual(len(bmu), 2)

    def test_bmu_in_range(self):
        """TC-K08: BMU в межах карти."""
        sample = np.array([0.5, 0.5])
        bmu = self.som._find_bmu(sample)
        self.assertTrue(0 <= bmu[0] < self.som.rows)
        self.assertTrue(0 <= bmu[1] < self.som.cols)

    def test_bmu_exact_match(self):
        """TC-K09: BMU для зразка, що збігається з вагою нейрона."""
        self.som.weights[1, 1] = np.array([0.5, 0.5])
        self.som.weights[0, 0] = np.array([0.0, 0.0])
        self.som.weights[2, 2] = np.array([1.0, 1.0])
        bmu = self.som._find_bmu(np.array([0.5, 0.5]))
        self.assertEqual(bmu, (1, 1))


class TestKohonenTraining(unittest.TestCase):
    """Тести навчання мережі."""

    def setUp(self):
        self.data = np.array([
            [0.1, 0.2], [0.15, 0.25], [0.12, 0.18],
            [0.8, 0.9], [0.85, 0.88], [0.82, 0.92],
        ])

    def test_normal_training(self):
        """TC-K10: Нормальне навчання з коректними даними."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2, random_seed=42)
        errors = som.train(self.data, epochs=10)
        self.assertEqual(len(errors), 10)
        self.assertTrue(som.is_trained)

    def test_training_error_decreases(self):
        """TC-K11: Помилка зменшується протягом навчання."""
        som = KohonenSOM(rows=5, cols=5, input_dim=2, learning_rate=0.5, radius=3.0)
        errors = som.train(self.data, epochs=50)
        self.assertLess(errors[-1], errors[0])

    def test_single_epoch_training(self):
        """TC-K12: Граничне значення — навчання 1 епоху."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2)
        errors = som.train(self.data, epochs=1)
        self.assertEqual(len(errors), 1)
        self.assertTrue(som.is_trained)

    def test_progress_callback(self):
        """TC-K13: Виклик callback під час навчання."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2)
        callback_calls = []

        def cb(epoch, total, error):
            callback_calls.append((epoch, total, error))

        som.train(self.data, epochs=5, progress_callback=cb)
        self.assertEqual(len(callback_calls), 5)

    def test_training_single_sample(self):
        """TC-K14: Навчання з одним зразком."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2)
        single = np.array([[0.5, 0.5]])
        errors = som.train(single, epochs=5)
        self.assertEqual(len(errors), 5)
        self.assertTrue(som.is_trained)


class TestKohonenClassification(unittest.TestCase):
    """Тести класифікації."""

    def setUp(self):
        self.som = KohonenSOM(rows=5, cols=5, input_dim=4, random_seed=42)
        np.random.seed(42)
        self.data = np.random.rand(30, 4)
        self.labels = np.array([0]*10 + [1]*10 + [2]*10)
        self.som.train(self.data, epochs=20)

    def test_classify_returns_correct_shape(self):
        """TC-K15: classify повертає масив правильної форми."""
        result = self.som.classify(self.data)
        self.assertEqual(result.shape, (30, 2))

    def test_hit_map_sum(self):
        """TC-K16: Сума hit map дорівнює кількості зразків."""
        hit_map = self.som.compute_hit_map(self.data)
        self.assertEqual(hit_map.sum(), 30)

    def test_hit_map_shape(self):
        """TC-K17: Форма hit map відповідає розміру карти."""
        hit_map = self.som.compute_hit_map(self.data)
        self.assertEqual(hit_map.shape, (5, 5))

    def test_umatrix_shape(self):
        """TC-K18: Форма U-matrix відповідає розміру карти."""
        umatrix = self.som.compute_umatrix()
        self.assertEqual(umatrix.shape, (5, 5))

    def test_umatrix_non_negative(self):
        """TC-K19: U-matrix містить лише невід'ємні значення."""
        umatrix = self.som.compute_umatrix()
        self.assertTrue(np.all(umatrix >= 0))

    def test_label_map_shape(self):
        """TC-K20: Форма label map відповідає розміру карти."""
        label_map = self.som.compute_label_map(self.data, self.labels, 3)
        self.assertEqual(label_map.shape, (5, 5))

    def test_label_map_values(self):
        """TC-K21: Label map містить лише -1 та валідні індекси класів."""
        label_map = self.som.compute_label_map(self.data, self.labels, 3)
        for val in label_map.flatten():
            self.assertIn(val, [-1, 0, 1, 2])

    def test_accuracy_range(self):
        """TC-K22: Точність у діапазоні [0, 1]."""
        accuracy = self.som.compute_accuracy(self.data, self.labels, 3)
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_predict_shape(self):
        """TC-K23: Predict повертає масив правильної довжини."""
        label_map = self.som.compute_label_map(self.data, self.labels, 3)
        predictions = self.som.predict(self.data, label_map)
        self.assertEqual(len(predictions), 30)


class TestKohonenEdgeCases(unittest.TestCase):
    """Тести виняткових та граничних ситуацій."""

    def test_zero_learning_rate(self):
        """TC-K24: Навчання з learning_rate=0 — ваги не змінюються (граничне)."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2, learning_rate=0.0)
        weights_before = som.weights.copy()
        data = np.array([[0.5, 0.5]])
        som.train(data, epochs=5)
        np.testing.assert_array_almost_equal(som.weights, weights_before, decimal=10)

    def test_classify_untrained(self):
        """TC-K25: Класифікація ненавченою мережею — не падає."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2)
        data = np.array([[0.5, 0.5]])
        result = som.classify(data)
        self.assertEqual(result.shape, (1, 2))

    def test_umatrix_1x1(self):
        """TC-K26: U-matrix для карти 1x1 — нуль."""
        som = KohonenSOM(rows=1, cols=1, input_dim=2)
        umatrix = som.compute_umatrix()
        self.assertEqual(umatrix[0, 0], 0)

    def test_hit_map_empty_data(self):
        """TC-K27: Hit map для порожнього масиву — нулі (виняткова ситуація)."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2)
        empty = np.empty((0, 2))
        hit_map = som.compute_hit_map(empty)
        self.assertEqual(hit_map.sum(), 0)

    def test_accuracy_no_valid_predictions(self):
        """TC-K28: Точність при відсутності валідних передбачень."""
        som = KohonenSOM(rows=3, cols=3, input_dim=2)
        data = np.empty((0, 2))
        labels = np.array([], dtype=int)
        accuracy = som.compute_accuracy(data, labels, 3)
        self.assertEqual(accuracy, 0.0)


if __name__ == "__main__":
    unittest.main()
