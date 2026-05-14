import os
import logging
import datetime
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

logger = logging.getLogger("kohonen_app")


class ReportGenerator:
    """Генерація графіків та текстових звітів."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_umatrix(self, umatrix: np.ndarray, save: bool = True) -> plt.Figure:
        """Побудувати U-matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(umatrix, cmap="bone_r", interpolation="nearest")
        ax.set_title("U-Matrix (відстані між сусідніми нейронами)", fontsize=13)
        ax.set_xlabel("Стовпець")
        ax.set_ylabel("Рядок")
        fig.colorbar(im, ax=ax, label="Середня відстань")
        fig.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "umatrix.png")
            fig.savefig(path, dpi=150)
            logger.info("U-matrix збережено: %s", path)
        return fig

    def plot_hit_map(self, hit_map: np.ndarray, save: bool = True) -> plt.Figure:
        """Побудувати Hit map."""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(hit_map, cmap="YlOrRd", interpolation="nearest")
        ax.set_title("Hit Map (кількість зразків на нейроні)", fontsize=13)
        ax.set_xlabel("Стовпець")
        ax.set_ylabel("Рядок")
        fig.colorbar(im, ax=ax, label="Кількість зразків")

        for i in range(hit_map.shape[0]):
            for j in range(hit_map.shape[1]):
                if hit_map[i, j] > 0:
                    ax.text(j, i, str(hit_map[i, j]),
                            ha="center", va="center", fontsize=7,
                            color="black" if hit_map[i, j] < hit_map.max() * 0.7 else "white")
        fig.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "hit_map.png")
            fig.savefig(path, dpi=150)
            logger.info("Hit map збережено: %s", path)
        return fig

    def plot_label_map(self, label_map: np.ndarray, target_names: list,
                       save: bool = True) -> plt.Figure:
        """Побудувати карту класів."""
        n_classes = len(target_names)
        colors = ["#e74c3c", "#2ecc71", "#3498db", "#f39c12", "#9b59b6"][:n_classes]
        colors_with_empty = ["#ecf0f1"] + colors
        cmap = ListedColormap(colors_with_empty)

        display_map = label_map.copy() + 1
        display_map[label_map == -1] = 0

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(display_map, cmap=cmap, interpolation="nearest",
                       vmin=0, vmax=n_classes)
        ax.set_title("Карта класів (домінуючий клас на нейроні)", fontsize=13)
        ax.set_xlabel("Стовпець")
        ax.set_ylabel("Рядок")

        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#ecf0f1", label="Порожній")]
        for i, name in enumerate(target_names):
            legend_elements.append(Patch(facecolor=colors[i], label=name))
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        fig.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "label_map.png")
            fig.savefig(path, dpi=150)
            logger.info("Карту класів збережено: %s", path)
        return fig

    def plot_training_error(self, errors: list, save: bool = True) -> plt.Figure:
        """Побудувати графік помилки навчання по епохах."""
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(errors) + 1), errors, color="#2c3e50", linewidth=1.5)
        ax.set_title("Помилка квантизації по епохах", fontsize=13)
        ax.set_xlabel("Епоха")
        ax.set_ylabel("Середня помилка квантизації")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        if save:
            path = os.path.join(self.output_dir, "training_error.png")
            fig.savefig(path, dpi=150)
            logger.info("Графік помилки збережено: %s", path)
        return fig

    def generate_text_report(self, params: dict, accuracy: float,
                             final_error: float, data_summary: str) -> str:
        """Згенерувати текстовий звіт та зберегти у файл."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "=" * 60,
            "ЗВІТ: Класифікація мережею Кохонена (SOM)",
            f"Дата: {now}",
            "=" * 60,
            "",
            "--- Дані ---",
            data_summary,
            "",
            "--- Параметри мережі ---",
            f"Розмір карти: {params.get('map_rows', '?')} x {params.get('map_cols', '?')}",
            f"Кількість епох: {params.get('epochs', '?')}",
            f"Початкова швидкість навчання: {params.get('learning_rate', '?')}",
            f"Початковий радіус: {params.get('radius', '?')}",
            "",
            "--- Результати ---",
            f"Фінальна помилка квантизації: {final_error:.6f}",
            f"Точність класифікації: {accuracy:.2%}",
            "",
            "--- Файли ---",
            f"U-matrix: {os.path.join(self.output_dir, 'umatrix.png')}",
            f"Hit map: {os.path.join(self.output_dir, 'hit_map.png')}",
            f"Карта класів: {os.path.join(self.output_dir, 'label_map.png')}",
            f"Графік помилки: {os.path.join(self.output_dir, 'training_error.png')}",
            "=" * 60,
        ]
        report_text = "\n".join(lines)

        path = os.path.join(self.output_dir, "report.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info("Текстовий звіт збережено: %s", path)

        return report_text
