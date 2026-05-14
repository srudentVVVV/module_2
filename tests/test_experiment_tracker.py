"""Unit-тести для модуля experiment_tracker.py — трекер експериментів."""

import csv
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.experiment_tracker import ExperimentTracker
from src.logger_setup import setup_logger

setup_logger("output/test.log")

PARAMS = {"map_rows": 10, "map_cols": 10, "epochs": 100, "learning_rate": 0.5, "radius": 5.0}


class TestExperimentTrackerNormal(unittest.TestCase):
    """Тести нормальних умов."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "experiments.json")
        self.tracker = ExperimentTracker(self.path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_add_experiment_returns_record(self):
        """TC-ET01: add_experiment повертає словник із коректними полями."""
        rec = self.tracker.add_experiment(PARAMS, 0.9533, 0.0234, 3.45)
        self.assertEqual(rec["id"], 1)
        self.assertIn("timestamp", rec)
        self.assertAlmostEqual(rec["accuracy"], 0.9533, places=4)
        self.assertAlmostEqual(rec["final_error"], 0.0234, places=4)
        self.assertAlmostEqual(rec["training_time"], 3.45, places=2)
        self.assertEqual(rec["params"], PARAMS)

    def test_add_multiple_experiments_increments_id(self):
        """TC-ET02: Кожен новий запис отримує унікальний зростаючий id."""
        for i in range(3):
            rec = self.tracker.add_experiment(PARAMS, 0.9 + i * 0.01, 0.05, 1.0)
            self.assertEqual(rec["id"], i + 1)

    def test_get_all_returns_all_records(self):
        """TC-ET03: get_all повертає стільки записів, скільки було додано."""
        for _ in range(4):
            self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.assertEqual(len(self.tracker.get_all()), 4)

    def test_get_all_returns_copy(self):
        """TC-ET04: get_all повертає копію — зміна результату не впливає на журнал."""
        self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        copy = self.tracker.get_all()
        copy.clear()
        self.assertEqual(len(self.tracker.get_all()), 1)

    def test_get_best_returns_highest_accuracy(self):
        """TC-ET05: get_best повертає запис з найвищою точністю."""
        self.tracker.add_experiment(PARAMS, 0.80, 0.05, 1.0)
        self.tracker.add_experiment(PARAMS, 0.95, 0.02, 1.0)
        self.tracker.add_experiment(PARAMS, 0.88, 0.03, 1.0)
        best = self.tracker.get_best()
        self.assertAlmostEqual(best["accuracy"], 0.95, places=4)
        self.assertEqual(best["id"], 2)

    def test_persistence_across_instances(self):
        """TC-ET06: Дані зберігаються у файл і відновлюються новим екземпляром."""
        self.tracker.add_experiment(PARAMS, 0.93, 0.04, 2.0)
        tracker2 = ExperimentTracker(self.path)
        records = tracker2.get_all()
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["accuracy"], 0.93, places=4)

    def test_params_stored_correctly(self):
        """TC-ET07: Параметри мережі зберігаються без змін."""
        custom = {"map_rows": 5, "map_cols": 7, "epochs": 50,
                  "learning_rate": 0.3, "radius": 3.0}
        rec = self.tracker.add_experiment(custom, 0.9, 0.05, 1.0)
        self.assertEqual(rec["params"]["map_rows"], 5)
        self.assertEqual(rec["params"]["map_cols"], 7)
        self.assertEqual(rec["params"]["learning_rate"], 0.3)

    def test_params_isolation(self):
        """TC-ET08: Зміна оригінального словника params після запису не впливає на журнал."""
        params = dict(PARAMS)
        self.tracker.add_experiment(params, 0.9, 0.05, 1.0)
        params["epochs"] = 9999
        stored = self.tracker.get_all()[0]["params"]["epochs"]
        self.assertEqual(stored, 100)

    def test_json_file_created(self):
        """TC-ET09: Файл JSON створюється після першого запису."""
        self.assertFalse(os.path.exists(self.path))
        self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.assertTrue(os.path.exists(self.path))

    def test_json_is_valid(self):
        """TC-ET10: Збережений файл є валідним JSON-масивом."""
        self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 1)


class TestExperimentTrackerExportCSV(unittest.TestCase):
    """Тести експорту у CSV."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "experiments.json")
        self.tracker = ExperimentTracker(self.path)
        self.csv_path = os.path.join(self.temp_dir, "export.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_export_csv_creates_file(self):
        """TC-ET11: export_csv створює файл."""
        self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.tracker.export_csv(self.csv_path)
        self.assertTrue(os.path.exists(self.csv_path))

    def test_export_csv_correct_row_count(self):
        """TC-ET12: CSV містить рядок заголовка + рядки для кожного запису."""
        for _ in range(3):
            self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.tracker.export_csv(self.csv_path)
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 4)  # 1 header + 3 data

    def test_export_csv_header_fields(self):
        """TC-ET13: CSV містить усі очікувані стовпці."""
        self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.tracker.export_csv(self.csv_path)
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        expected = {"id", "timestamp", "map_rows", "map_cols", "epochs",
                    "learning_rate", "radius", "accuracy", "final_error", "training_time"}
        self.assertEqual(set(header), expected)

    def test_export_csv_values(self):
        """TC-ET14: CSV містить коректні числові значення."""
        self.tracker.add_experiment(PARAMS, 0.9533, 0.0234, 3.45)
        self.tracker.export_csv(self.csv_path)
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        self.assertAlmostEqual(float(row["accuracy"]), 0.9533, places=4)
        self.assertEqual(int(row["epochs"]), 100)

    def test_export_empty_tracker(self):
        """TC-ET15: export_csv з порожнім журналом — файл лише з заголовком."""
        self.tracker.export_csv(self.csv_path)
        with open(self.csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 1)


class TestExperimentTrackerClear(unittest.TestCase):
    """Тести очищення журналу."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "experiments.json")
        self.tracker = ExperimentTracker(self.path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_clear_removes_all_records(self):
        """TC-ET16: clear видаляє всі записи з пам'яті."""
        for _ in range(3):
            self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.tracker.clear()
        self.assertEqual(len(self.tracker.get_all()), 0)

    def test_clear_persists_to_file(self):
        """TC-ET17: clear зберігає порожній масив у файл."""
        self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.tracker.clear()
        tracker2 = ExperimentTracker(self.path)
        self.assertEqual(len(tracker2.get_all()), 0)

    def test_add_after_clear_resets_id(self):
        """TC-ET18: Після clear новий запис отримує id=1."""
        self.tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.tracker.clear()
        rec = self.tracker.add_experiment(PARAMS, 0.88, 0.06, 1.5)
        self.assertEqual(rec["id"], 1)


class TestExperimentTrackerEdgeCases(unittest.TestCase):
    """Граничні та виняткові умови."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.temp_dir, "experiments.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_best_empty_returns_none(self):
        """TC-ET19: get_best на порожньому журналі повертає None."""
        tracker = ExperimentTracker(self.path)
        self.assertIsNone(tracker.get_best())

    def test_load_corrupted_json(self):
        """TC-ET20: Пошкоджений JSON — журнал стартує порожнім."""
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not valid json!!!")
        tracker = ExperimentTracker(self.path)
        self.assertEqual(len(tracker.get_all()), 0)

    def test_load_non_list_json(self):
        """TC-ET21: JSON не є масивом — журнал стартує порожнім."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
        tracker = ExperimentTracker(self.path)
        self.assertEqual(len(tracker.get_all()), 0)

    def test_load_missing_file(self):
        """TC-ET22: Файл відсутній — журнал стартує порожнім без помилок."""
        tracker = ExperimentTracker(self.path)
        self.assertEqual(len(tracker.get_all()), 0)

    def test_accuracy_zero(self):
        """TC-ET23: Граничне значення точності = 0."""
        tracker = ExperimentTracker(self.path)
        rec = tracker.add_experiment(PARAMS, 0.0, 1.0, 1.0)
        self.assertEqual(rec["accuracy"], 0.0)

    def test_accuracy_one(self):
        """TC-ET24: Граничне значення точності = 1."""
        tracker = ExperimentTracker(self.path)
        rec = tracker.add_experiment(PARAMS, 1.0, 0.0, 1.0)
        self.assertEqual(rec["accuracy"], 1.0)

    def test_nested_dir_creation(self):
        """TC-ET25: Файл у неіснуючому підкаталозі — каталог створюється автоматично."""
        nested_path = os.path.join(self.temp_dir, "a", "b", "experiments.json")
        tracker = ExperimentTracker(nested_path)
        tracker.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        self.assertTrue(os.path.exists(nested_path))

    def test_get_best_single_record(self):
        """TC-ET26: get_best з одним записом повертає цей запис."""
        tracker = ExperimentTracker(self.path)
        rec = tracker.add_experiment(PARAMS, 0.75, 0.1, 2.0)
        best = tracker.get_best()
        self.assertEqual(best["id"], rec["id"])

    def test_training_time_zero(self):
        """TC-ET27: Граничне значення часу навчання = 0."""
        tracker = ExperimentTracker(self.path)
        rec = tracker.add_experiment(PARAMS, 0.9, 0.05, 0.0)
        self.assertEqual(rec["training_time"], 0.0)

    def test_load_failed_flag_on_corrupted_json(self):
        """TC-ET28: BUG-ET-001 — load_failed=True при пошкодженому JSON."""
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("{not valid json!!!")
        tracker = ExperimentTracker(self.path)
        self.assertTrue(tracker.load_failed,
                        "load_failed має бути True при пошкодженому JSON")

    def test_load_failed_flag_on_non_list_json(self):
        """TC-ET29: load_failed=True коли JSON не є масивом."""
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"key": "value"}, f)
        tracker = ExperimentTracker(self.path)
        self.assertTrue(tracker.load_failed)

    def test_load_failed_false_on_valid_file(self):
        """TC-ET30: load_failed=False при коректному файлі."""
        tracker1 = ExperimentTracker(self.path)
        tracker1.add_experiment(PARAMS, 0.9, 0.05, 1.0)
        tracker2 = ExperimentTracker(self.path)
        self.assertFalse(tracker2.load_failed)

    def test_load_failed_false_on_missing_file(self):
        """TC-ET31: load_failed=False коли файл відсутній (не помилка)."""
        tracker = ExperimentTracker(self.path)
        self.assertFalse(tracker.load_failed)


if __name__ == "__main__":
    unittest.main()
