import csv
import json
import logging
import os
import datetime
from typing import Optional

logger = logging.getLogger("kohonen_app")


class ExperimentTracker:
    """Зберігає результати кожного навчального запуску у JSON-файл."""

    def __init__(self, storage_path: str = "output/experiments.json"):
        self.storage_path = storage_path
        self.load_failed: bool = False
        self._records: list = self._load()

    def _load(self) -> list:
        if not os.path.exists(self.storage_path):
            return []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Журнал містить некоректний формат: %s", self.storage_path)
                self.load_failed = True
                return []
            return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Не вдалося завантажити журнал: %s", self.storage_path)
            self.load_failed = True
            return []

    def _save(self) -> None:
        dir_path = os.path.dirname(self.storage_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def add_experiment(self, params: dict, accuracy: float,
                       final_error: float, training_time: float) -> dict:
        """Записати результат навчання. Повертає збережений запис."""
        record = {
            "id": len(self._records) + 1,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "params": dict(params),
            "accuracy": round(float(accuracy), 6),
            "final_error": round(float(final_error), 6),
            "training_time": round(float(training_time), 2),
        }
        self._records.append(record)
        self._save()
        logger.info("Експеримент #%d збережено: точність=%.4f, помилка=%.6f",
                    record["id"], accuracy, final_error)
        return record

    def get_all(self) -> list:
        """Повернути копію всіх записів."""
        return list(self._records)

    def get_best(self) -> Optional[dict]:
        """Повернути запис з найвищою точністю."""
        if not self._records:
            return None
        return max(self._records, key=lambda r: r["accuracy"])

    def export_csv(self, path: str) -> None:
        """Зберегти всі записи у CSV-файл."""
        fieldnames = [
            "id", "timestamp",
            "map_rows", "map_cols", "epochs", "learning_rate", "radius",
            "accuracy", "final_error", "training_time",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in self._records:
                row = {
                    "id": rec["id"],
                    "timestamp": rec["timestamp"],
                    **rec["params"],
                    "accuracy": rec["accuracy"],
                    "final_error": rec["final_error"],
                    "training_time": rec["training_time"],
                }
                writer.writerow(row)
        logger.info("Журнал експортовано у CSV: %s", path)

    def clear(self) -> None:
        """Видалити всі записи та очистити файл."""
        self._records.clear()
        self._save()
        logger.info("Журнал експериментів очищено")
