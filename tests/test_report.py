"""Unit-тести для модуля report.py — генерація звітів."""

import unittest
import os
import sys
import tempfile
import shutil
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.report import ReportGenerator
from src.logger_setup import setup_logger

setup_logger("output/test.log")


class TestReportGeneratorNormal(unittest.TestCase):
    """Тести нормальних умов для ReportGenerator."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rg = ReportGenerator(self.temp_dir)

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.temp_dir)

    def test_plot_umatrix_returns_figure(self):
        """TC-R01: plot_umatrix повертає matplotlib Figure."""
        umatrix = np.random.rand(5, 5)
        fig = self.rg.plot_umatrix(umatrix, save=True)
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "umatrix.png")))

    def test_plot_hit_map_returns_figure(self):
        """TC-R02: plot_hit_map повертає matplotlib Figure."""
        hit_map = np.array([[5, 0, 3], [0, 10, 0], [2, 0, 1]])
        fig = self.rg.plot_hit_map(hit_map, save=True)
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "hit_map.png")))

    def test_plot_training_error_returns_figure(self):
        """TC-R03: plot_training_error повертає matplotlib Figure."""
        errors = [0.5, 0.4, 0.3, 0.2, 0.1]
        fig = self.rg.plot_training_error(errors, save=True)
        self.assertIsInstance(fig, plt.Figure)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "training_error.png")))

    def test_generate_text_report(self):
        """TC-R04: Текстовий звіт генерується та зберігається."""
        params = {"map_rows": 5, "map_cols": 5, "epochs": 10,
                  "learning_rate": 0.5, "radius": 3.0}
        report = self.rg.generate_text_report(params, 0.95, 0.05, "Test data")
        self.assertIn("95.00%", report)
        self.assertIn("0.050000", report)
        report_path = os.path.join(self.temp_dir, "report.txt")
        self.assertTrue(os.path.exists(report_path))

    def test_plot_label_map_returns_figure(self):
        """TC-R05: plot_label_map повертає matplotlib Figure."""
        label_map = np.array([[0, 1, -1], [2, 0, 1], [-1, -1, 2]])
        fig = self.rg.plot_label_map(label_map, ["a", "b", "c"], save=True)
        self.assertIsInstance(fig, plt.Figure)


class TestReportGeneratorEdgeCases(unittest.TestCase):
    """Тести граничних та виняткових умов."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.rg = ReportGenerator(self.temp_dir)

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.temp_dir)

    def test_plot_training_error_empty(self):
        """TC-R06: Графік помилки з порожнім списком — не падає."""
        fig = self.rg.plot_training_error([], save=False)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_umatrix_1x1(self):
        """TC-R07: U-matrix розміром 1x1."""
        umatrix = np.array([[0.0]])
        fig = self.rg.plot_umatrix(umatrix, save=False)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_hit_map_all_zeros(self):
        """TC-R08: Hit map з усіма нулями."""
        hit_map = np.zeros((3, 3), dtype=int)
        fig = self.rg.plot_hit_map(hit_map, save=False)
        self.assertIsInstance(fig, plt.Figure)

    def test_plot_label_map_all_empty(self):
        """TC-R09: Карта класів без жодного зразка (всі -1)."""
        label_map = np.full((3, 3), -1)
        fig = self.rg.plot_label_map(label_map, ["a", "b", "c"], save=False)
        self.assertIsInstance(fig, plt.Figure)

    def test_label_map_color_count(self):
        """TC-R10: BUG — Кількість кольорів у colormap повинна відповідати n_classes+1.
        Перевіряє, чи карта класів використовує правильну кількість кольорів."""
        label_map = np.array([[0, 1, 2], [0, -1, 1], [2, 2, 0]])
        target_names = ["setosa", "versicolor", "virginica"]
        fig = self.rg.plot_label_map(label_map, target_names, save=False)
        ax = fig.axes[0]
        im = ax.images[0]
        cmap = im.cmap
        # n_classes = 3, colormap повинна мати рівно 4 кольори (empty + 3 класи)
        self.assertEqual(
            cmap.N, len(target_names) + 1,
            f"BUG: colormap має {cmap.N} кольорів замість {len(target_names) + 1}. "
            f"Зайві кольори спотворюють візуалізацію!"
        )

    def test_generate_report_missing_params(self):
        """TC-R11: Звіт з неповними параметрами — підставляє '?'."""
        report = self.rg.generate_text_report({}, 0.5, 0.1, "No data")
        self.assertIn("?", report)

    def test_output_dir_creation(self):
        """TC-R12: Автоматичне створення output каталогу."""
        new_dir = os.path.join(self.temp_dir, "nested", "output")
        rg = ReportGenerator(new_dir)
        self.assertTrue(os.path.isdir(new_dir))


if __name__ == "__main__":
    unittest.main()
