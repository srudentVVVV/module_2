import numpy as np
import logging

logger = logging.getLogger("kohonen_app")


class KohonenSOM:
    """Самоорганізаційна карта Кохонена (Self-Organizing Map)."""

    def __init__(self, rows: int, cols: int, input_dim: int,
                 learning_rate: float = 0.5, radius: float = 5.0,
                 random_seed: int = 42):
        self.rows = rows
        self.cols = cols
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.initial_radius = radius
        self.random_seed = random_seed

        self.weights = None
        self.training_errors = []
        self.is_trained = False

        self._init_weights()

    def _init_weights(self) -> None:
        """Ініціалізація ваг випадковими значеннями [0, 1]."""
        rng = np.random.default_rng(self.random_seed)
        self.weights = rng.random((self.rows, self.cols, self.input_dim))
        logger.info(
            "Ваги ініціалізовано: карта %dx%d, вхідна розмірність %d",
            self.rows, self.cols, self.input_dim,
        )

    def _find_bmu(self, sample: np.ndarray) -> tuple:
        """Знайти Best Matching Unit (нейрон-переможець) для зразка."""
        distances = np.linalg.norm(self.weights - sample, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx

    def _decay_learning_rate(self, epoch: int, total_epochs: int) -> float:
        """Затухання швидкості навчання."""
        return self.initial_learning_rate * np.exp(-epoch / total_epochs)

    def _decay_radius(self, epoch: int, total_epochs: int) -> float:
        """Затухання радіусу сусідства."""
        return self.initial_radius * np.exp(-epoch / (total_epochs / np.log(self.initial_radius + 1e-8)))

    def _neighborhood(self, bmu: tuple, radius: float) -> np.ndarray:
        """Обчислити функцію сусідства (Гаусова) для всіх нейронів."""
        row_indices, col_indices = np.meshgrid(
            np.arange(self.rows), np.arange(self.cols), indexing="ij"
        )
        dist_sq = (row_indices - bmu[0]) ** 2 + (col_indices - bmu[1]) ** 2
        return np.exp(-dist_sq / (2 * (radius ** 2 + 1e-8)))

    def train(self, data: np.ndarray, epochs: int,
              progress_callback=None) -> list:
        """
        Навчання мережі Кохонена.

        Args:
            data: масив даних (n_samples, n_features)
            epochs: кількість епох
            progress_callback: функція зворотного виклику (epoch, total, error)

        Returns:
            Список помилок квантизації по епохах.
        """
        self.training_errors = []
        n_samples = data.shape[0]

        logger.info(
            "Початок навчання: %d епох, %d зразків, lr=%.4f, radius=%.2f",
            epochs, n_samples, self.initial_learning_rate, self.initial_radius,
        )

        for epoch in range(epochs):
            lr = self._decay_learning_rate(epoch, epochs)
            radius = self._decay_radius(epoch, epochs)

            indices = np.random.permutation(n_samples)
            epoch_error = 0.0

            for idx in indices:
                sample = data[idx]
                bmu = self._find_bmu(sample)

                influence = self._neighborhood(bmu, radius)
                influence = influence[:, :, np.newaxis]

                diff = sample - self.weights
                self.weights += lr * influence * diff

                epoch_error += np.linalg.norm(sample - self.weights[bmu])

            avg_error = epoch_error / n_samples
            self.training_errors.append(avg_error)

            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_error)

            if (epoch + 1) % max(1, epochs // 10) == 0:
                logger.info(
                    "Епоха %d/%d — помилка: %.6f, lr: %.6f, radius: %.4f",
                    epoch + 1, epochs, avg_error, lr, radius,
                )

        self.is_trained = True
        logger.info(
            "Навчання завершено. Фінальна помилка: %.6f",
            self.training_errors[-1] if self.training_errors else 0,
        )
        return self.training_errors

    def classify(self, data: np.ndarray) -> np.ndarray:
        """Класифікувати дані: повернути індекси BMU для кожного зразка."""
        if not self.is_trained:
            logger.warning("Мережа не навчена. Класифікація може бути неточною.")
        bmu_indices = np.array([self._find_bmu(sample) for sample in data])
        return bmu_indices

    def compute_umatrix(self) -> np.ndarray:
        """Обчислити U-matrix (середня відстань до сусідів)."""
        umatrix = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            dist = np.linalg.norm(
                                self.weights[i, j] - self.weights[ni, nj]
                            )
                            neighbors.append(dist)
                umatrix[i, j] = np.mean(neighbors) if neighbors else 0
        return umatrix

    def compute_hit_map(self, data: np.ndarray) -> np.ndarray:
        """Обчислити hit map — кількість зразків, що потрапили на кожен нейрон."""
        hit_map = np.zeros((self.rows, self.cols), dtype=int)
        for sample in data:
            bmu = self._find_bmu(sample)
            hit_map[bmu] += 1
        return hit_map

    def compute_label_map(self, data: np.ndarray, labels: np.ndarray,
                          n_classes: int) -> np.ndarray:
        """Створити карту міток: для кожного нейрона визначити домінуючий клас."""
        class_counts = np.zeros((self.rows, self.cols, n_classes), dtype=int)
        for sample, label in zip(data, labels):
            bmu = self._find_bmu(sample)
            class_counts[bmu[0], bmu[1], label] += 1

        label_map = np.full((self.rows, self.cols), -1, dtype=int)
        for i in range(self.rows):
            for j in range(self.cols):
                if class_counts[i, j].sum() > 0:
                    label_map[i, j] = np.argmax(class_counts[i, j])
        return label_map

    def predict(self, data: np.ndarray, label_map: np.ndarray) -> np.ndarray:
        """Передбачити клас для кожного зразка на основі label_map."""
        predictions = []
        for sample in data:
            bmu = self._find_bmu(sample)
            predictions.append(label_map[bmu])
        return np.array(predictions)

    def compute_accuracy(self, data: np.ndarray, labels: np.ndarray,
                         n_classes: int) -> float:
        """Обчислити точність класифікації."""
        label_map = self.compute_label_map(data, labels, n_classes)
        predictions = self.predict(data, label_map)
        valid_mask = predictions >= 0
        if valid_mask.sum() == 0:
            return 0.0
        accuracy = np.mean(predictions[valid_mask] == labels[valid_mask])
        return float(accuracy)
