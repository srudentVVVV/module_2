import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """Завантаження та підготовка даних для мережі Кохонена."""

    def __init__(self):
        self.raw_data = None
        self.normalized_data = None
        self.labels = None
        self.feature_names = None
        self.target_names = None
        self.scaler = MinMaxScaler()

    def load_iris(self) -> dict:
        """Завантажити датасет Iris та нормалізувати."""
        iris = load_iris()
        self.raw_data = iris.data.astype(np.float64)
        self.labels = iris.target
        self.feature_names = list(iris.feature_names)
        self.target_names = list(iris.target_names)

        self.normalized_data = self.scaler.fit_transform(self.raw_data)

        return {
            "data": self.normalized_data,
            "labels": self.labels,
            "feature_names": self.feature_names,
            "target_names": self.target_names,
            "n_samples": self.raw_data.shape[0],
            "n_features": self.raw_data.shape[1],
            "n_classes": len(self.target_names),
        }

    def get_summary(self) -> str:
        """Повернути текстовий опис завантажених даних."""
        if self.raw_data is None:
            return "Дані не завантажені."
        lines = [
            f"Датасет: Iris",
            f"Кількість зразків: {self.raw_data.shape[0]}",
            f"Кількість ознак: {self.raw_data.shape[1]}",
            f"Ознаки: {', '.join(self.feature_names)}",
            f"Класи: {', '.join(self.target_names)}",
            f"Розподіл класів: {dict(zip([str(n) for n in self.target_names], [int(c) for c in np.bincount(self.labels)]))}",
        ]
        return "\n".join(lines)
