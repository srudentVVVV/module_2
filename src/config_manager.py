import copy
import json
import os


DEFAULT_CONFIG = {
    "version": "1.0.0",
    "language": "uk",
    "theme_color": "#2c3e50",
    "accent_color": "#3498db",
    "output_dir": "output",
    "log_file": "output/app.log",
    "default_params": {
        "map_rows": 10,
        "map_cols": 10,
        "epochs": 100,
        "learning_rate": 0.5,
        "radius": 5.0,
    },
}


class ConfigManager:
    """Менеджер конфігурації: читання/запис config.json."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = {}
        self.load()

    def load(self) -> dict:
        """Завантажити конфігурацію з файлу або створити за замовчуванням."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.config = copy.deepcopy(DEFAULT_CONFIG)
                self.save()
        else:
            self.config = copy.deepcopy(DEFAULT_CONFIG)
            self.save()
        return self.config

    def save(self) -> None:
        """Зберегти поточну конфігурацію у файл."""
        config_dir = os.path.dirname(self.config_path)
        if config_dir and not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default=None):
        """Отримати значення за ключем."""
        return self.config.get(key, default)

    def set(self, key: str, value) -> None:
        """Встановити значення та зберегти."""
        self.config[key] = value
        self.save()

    def get_default_params(self) -> dict:
        """Повернути параметри SOM за замовчуванням."""
        return self.config.get("default_params", DEFAULT_CONFIG["default_params"])
