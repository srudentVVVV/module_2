"""
Класифікація об'єктів за допомогою мережі Кохонена (SOM).
Точка входу програми.
"""

import os
import sys
import tkinter as tk

# Встановити робочий каталог на каталог скрипта
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.config_manager import ConfigManager
from src.logger_setup import setup_logger
from src.gui import KohonenApp


def main():
    config = ConfigManager("config.json")

    logger = setup_logger(config.get("log_file", "output/app.log"))
    logger.info("=" * 50)
    logger.info("Запуск програми v%s", config.get("version", "1.0.0"))

    try:
        root = tk.Tk()
        app = KohonenApp(root, config)
        logger.info("GUI ініціалізовано")
        root.mainloop()
    except Exception as e:
        logger.critical("Критична помилка: %s", e, exc_info=True)
        print(f"Критична помилка: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        logger.info("Програму завершено")


if __name__ == "__main__":
    main()
