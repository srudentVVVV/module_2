"""Unit-тести для модуля config_manager.py — менеджер конфігурації."""

import unittest
import os
import sys
import json
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config_manager import ConfigManager, DEFAULT_CONFIG


class TestConfigManagerNormal(unittest.TestCase):
    """Тести нормальних умов для ConfigManager."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_default_config(self):
        """TC-C01: Створення конфігурації за замовчуванням при відсутності файлу."""
        cm = ConfigManager(self.config_path)
        self.assertTrue(os.path.exists(self.config_path))

    def test_load_existing_config(self):
        """TC-C02: Завантаження існуючої конфігурації."""
        data = {"version": "2.0.0", "language": "en", "default_params": {"map_rows": 5}}
        with open(self.config_path, "w") as f:
            json.dump(data, f)
        cm = ConfigManager(self.config_path)
        self.assertEqual(cm.get("version"), "2.0.0")

    def test_get_existing_key(self):
        """TC-C03: Отримання значення існуючого ключа."""
        cm = ConfigManager(self.config_path)
        self.assertEqual(cm.get("version"), "1.0.0")

    def test_get_missing_key_default(self):
        """TC-C04: Отримання неіснуючого ключа — повертає default."""
        cm = ConfigManager(self.config_path)
        self.assertEqual(cm.get("nonexistent", "fallback"), "fallback")

    def test_set_and_persist(self):
        """TC-C05: Встановлення значення зберігається у файл."""
        cm = ConfigManager(self.config_path)
        cm.set("language", "en")
        cm2 = ConfigManager(self.config_path)
        self.assertEqual(cm2.get("language"), "en")

    def test_get_default_params(self):
        """TC-C06: Отримання параметрів SOM за замовчуванням."""
        cm = ConfigManager(self.config_path)
        params = cm.get_default_params()
        self.assertIn("map_rows", params)
        self.assertIn("learning_rate", params)


class TestConfigManagerEdgeCases(unittest.TestCase):
    """Тести граничних та виняткових умов."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_corrupted_json(self):
        """TC-C07: Завантаження пошкодженого JSON — відновлення за замовчуванням."""
        with open(self.config_path, "w") as f:
            f.write("{invalid json!!!")
        cm = ConfigManager(self.config_path)
        self.assertEqual(cm.get("version"), DEFAULT_CONFIG["version"])

    def test_empty_file(self):
        """TC-C08: Завантаження порожнього файлу — відновлення за замовчуванням."""
        with open(self.config_path, "w") as f:
            f.write("")
        cm = ConfigManager(self.config_path)
        self.assertEqual(cm.get("version"), DEFAULT_CONFIG["version"])

    def test_nested_dir_creation(self):
        """TC-C09: Створення конфігурації у неіснуючому підкаталозі."""
        nested_path = os.path.join(self.temp_dir, "a", "b", "config.json")
        cm = ConfigManager(nested_path)
        self.assertTrue(os.path.exists(nested_path))

    def test_default_config_isolation(self):
        """TC-C10: BUG — Зміна config не повинна впливати на DEFAULT_CONFIG.
        Перевіряє, чи shallow copy не мутує глобальний DEFAULT_CONFIG."""
        cm = ConfigManager(self.config_path)
        original_rows = DEFAULT_CONFIG["default_params"]["map_rows"]
        cm.config["default_params"]["map_rows"] = 999
        self.assertEqual(
            DEFAULT_CONFIG["default_params"]["map_rows"],
            original_rows,
            "BUG: shallow copy мутує DEFAULT_CONFIG — потрібен deepcopy!"
        )


if __name__ == "__main__":
    unittest.main()
