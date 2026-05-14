import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from src.config_manager import ConfigManager
from src.data_loader import DataLoader
from src.kohonen import KohonenSOM
from src.report import ReportGenerator
from src.logger_setup import setup_logger

logger = logging.getLogger("kohonen_app")


class ToolTip:
    """Підказка при наведенні мишкою на віджет."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip_window:
            return
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 25
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT,
            background="#ffffcc", relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9),
            wraplength=300,
        )
        label.pack(ipadx=4, ipady=2)

    def _hide(self, event=None):
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None


class KohonenApp:
    """Головне вікно програми класифікації мережею Кохонена."""

    def __init__(self, root: tk.Tk, config: ConfigManager):
        self.root = root
        self.config = config
        self.data_loader = DataLoader()
        self.som = None
        self.report_gen = ReportGenerator(config.get("output_dir", "output"))
        self.data_info = None
        self.is_training = False

        self._current_figures = []

        self._setup_window()
        self._create_menu()
        self._create_widgets()
        self._load_data()

    def _setup_window(self):
        """Налаштування головного вікна."""
        self.root.title(
            f"Класифікація мережею Кохонена v{self.config.get('version', '1.0.0')}"
        )
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        theme_color = self.config.get("theme_color", "#2c3e50")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f5f6fa")
        style.configure("TLabel", background="#f5f6fa", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure(
            "Header.TLabel", background="#f5f6fa",
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Accent.TButton", font=("Segoe UI", 10, "bold"),
        )
        style.configure("TLabelframe", background="#f5f6fa")
        style.configure("TLabelframe.Label", background="#f5f6fa", font=("Segoe UI", 10, "bold"))

    def _create_menu(self):
        """Створити меню."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Зберегти звіт", command=self._save_report)
        file_menu.add_separator()
        file_menu.add_command(label="Вихід", command=self.root.quit)
        menubar.add_cascade(label="Файл", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Про програму", command=self._show_about)
        menubar.add_cascade(label="Довідка", menu=help_menu)

    def _create_widgets(self):
        """Створити всі віджети."""
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_paned, width=320)
        main_paned.add(left_frame, weight=0)

        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        self._create_params_panel(left_frame)
        self._create_buttons_panel(left_frame)
        self._create_data_info_panel(left_frame)
        self._create_log_panel(left_frame)

        self._create_visualization_panel(right_frame)

        self._create_status_bar()

    def _create_params_panel(self, parent):
        """Панель параметрів SOM."""
        frame = ttk.LabelFrame(parent, text="Параметри мережі Кохонена")
        frame.pack(fill=tk.X, padx=5, pady=5)

        params = self.config.get_default_params()

        self.param_vars = {}
        fields = [
            ("map_rows", "Рядки карти:", params.get("map_rows", 10),
             "Кількість рядків у карті SOM (1-50)"),
            ("map_cols", "Стовпці карти:", params.get("map_cols", 10),
             "Кількість стовпців у карті SOM (1-50)"),
            ("epochs", "Кількість епох:", params.get("epochs", 100),
             "Скільки разів мережа пройде по всіх даних (1-10000)"),
            ("learning_rate", "Швидкість навчання:", params.get("learning_rate", 0.5),
             "Початкова швидкість навчання (0.001-1.0). Зменшується під час навчання"),
            ("radius", "Радіус сусідства:", params.get("radius", 5.0),
             "Початковий радіус впливу BMU (0.1-50.0). Зменшується під час навчання"),
        ]

        for i, (key, label_text, default, tooltip) in enumerate(fields):
            lbl = ttk.Label(frame, text=label_text)
            lbl.grid(row=i, column=0, sticky=tk.W, padx=5, pady=3)

            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(frame, textvariable=var, width=12)
            entry.grid(row=i, column=1, padx=5, pady=3)

            self.param_vars[key] = var
            ToolTip(entry, tooltip)
            ToolTip(lbl, tooltip)

        frame.columnconfigure(1, weight=1)

    def _create_buttons_panel(self, parent):
        """Панель кнопок управління."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.btn_train = ttk.Button(
            frame, text="Навчити мережу", command=self._start_training,
            style="Accent.TButton",
        )
        self.btn_train.pack(fill=tk.X, pady=2)
        ToolTip(self.btn_train, "Розпочати навчання мережі Кохонена з вказаними параметрами")

        self.btn_classify = ttk.Button(
            frame, text="Класифікувати", command=self._classify,
            state=tk.DISABLED,
        )
        self.btn_classify.pack(fill=tk.X, pady=2)
        ToolTip(self.btn_classify, "Класифікувати дані та показати результати")

        self.btn_report = ttk.Button(
            frame, text="Зберегти звіт", command=self._save_report,
            state=tk.DISABLED,
        )
        self.btn_report.pack(fill=tk.X, pady=2)
        ToolTip(self.btn_report, "Зберегти графіки та текстовий звіт у каталог output/")

        self.btn_reset = ttk.Button(
            frame, text="Скинути", command=self._reset,
        )
        self.btn_reset.pack(fill=tk.X, pady=2)
        ToolTip(self.btn_reset, "Скинути мережу та результати")

    def _create_data_info_panel(self, parent):
        """Панель інформації про дані."""
        frame = ttk.LabelFrame(parent, text="Інформація про дані")
        frame.pack(fill=tk.X, padx=5, pady=5)

        self.data_info_label = ttk.Label(frame, text="Завантаження...", wraplength=280)
        self.data_info_label.pack(padx=5, pady=5, anchor=tk.W)

    def _create_log_panel(self, parent):
        """Панель логу подій."""
        frame = ttk.LabelFrame(parent, text="Журнал подій")
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            frame, height=8, wrap=tk.WORD, font=("Consolas", 9),
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=3, pady=3)

    def _create_visualization_panel(self, parent):
        """Панель візуалізації з вкладками."""
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_umatrix = ttk.Frame(self.notebook)
        self.tab_hitmap = ttk.Frame(self.notebook)
        self.tab_labelmap = ttk.Frame(self.notebook)
        self.tab_error = ttk.Frame(self.notebook)
        self.tab_results = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_umatrix, text="U-Matrix")
        self.notebook.add(self.tab_hitmap, text="Hit Map")
        self.notebook.add(self.tab_labelmap, text="Карта класів")
        self.notebook.add(self.tab_error, text="Графік помилки")
        self.notebook.add(self.tab_results, text="Результати")

        self.results_text = scrolledtext.ScrolledText(
            self.tab_results, wrap=tk.WORD, font=("Consolas", 10),
            state=tk.DISABLED,
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        placeholder_text = "Натисніть «Навчити мережу» для початку роботи"
        for tab in [self.tab_umatrix, self.tab_hitmap, self.tab_labelmap, self.tab_error]:
            lbl = ttk.Label(tab, text=placeholder_text, font=("Segoe UI", 12))
            lbl.pack(expand=True)

    def _create_status_bar(self):
        """Статус-бар внизу вікна."""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Готово")
        self.status_label = ttk.Label(
            status_frame, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(5, 2),
        )
        self.status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

        self.progress = ttk.Progressbar(
            status_frame, mode="determinate", length=200,
        )
        self.progress.pack(side=tk.RIGHT, padx=5, pady=2)

    def _load_data(self):
        """Завантажити дані Iris."""
        try:
            self.data_info = self.data_loader.load_iris()
            summary = self.data_loader.get_summary()
            self.data_info_label.config(text=summary)
            self._log("Дані Iris завантажено успішно.")
            logger.info("Дані Iris завантажено: %d зразків", self.data_info["n_samples"])
        except Exception as e:
            msg = f"Помилка завантаження даних: {e}"
            self.data_info_label.config(text=msg)
            self._log(msg)
            logger.error(msg, exc_info=True)
            messagebox.showerror("Помилка", msg)

    def _validate_params(self) -> dict | None:
        """Валідація параметрів введених користувачем."""
        try:
            rows = int(self.param_vars["map_rows"].get())
            cols = int(self.param_vars["map_cols"].get())
            epochs = int(self.param_vars["epochs"].get())
            lr = float(self.param_vars["learning_rate"].get())
            radius = float(self.param_vars["radius"].get())
        except ValueError:
            messagebox.showerror(
                "Помилка введення",
                "Усі параметри повинні бути числами.\n"
                "Рядки, стовпці, епохи — цілі числа.\n"
                "Швидкість навчання та радіус — дробові.",
            )
            return None

        errors = []
        if not (1 <= rows <= 50):
            errors.append("Рядки карти: від 1 до 50")
        if not (1 <= cols <= 50):
            errors.append("Стовпці карти: від 1 до 50")
        if not (1 <= epochs <= 10000):
            errors.append("Кількість епох: від 1 до 10000")
        if not (0.001 <= lr <= 1.0):
            errors.append("Швидкість навчання: від 0.001 до 1.0")
        if not (0.1 <= radius <= 50.0):
            errors.append("Радіус сусідства: від 0.1 до 50.0")

        if errors:
            messagebox.showerror(
                "Помилка валідації",
                "Невірні значення параметрів:\n\n" + "\n".join(f"• {e}" for e in errors),
            )
            return None

        return {
            "map_rows": rows,
            "map_cols": cols,
            "epochs": epochs,
            "learning_rate": lr,
            "radius": radius,
        }

    def _start_training(self):
        """Розпочати навчання у фоновому потоці."""
        if self.is_training:
            messagebox.showwarning("Увага", "Навчання вже виконується.")
            return

        if self.data_info is None:
            messagebox.showerror("Помилка", "Дані не завантажені.")
            return

        params = self._validate_params()
        if params is None:
            return

        self.is_training = True
        self.btn_train.config(state=tk.DISABLED)
        self.btn_classify.config(state=tk.DISABLED)
        self.btn_report.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self._log(f"Початок навчання: {params}")
        self.status_var.set("Навчання...")

        self.current_params = params

        thread = threading.Thread(target=self._train_thread, args=(params,), daemon=True)
        thread.start()

    def _train_thread(self, params: dict):
        """Потік навчання."""
        try:
            self.som = KohonenSOM(
                rows=params["map_rows"],
                cols=params["map_cols"],
                input_dim=self.data_info["n_features"],
                learning_rate=params["learning_rate"],
                radius=params["radius"],
            )

            def progress_cb(epoch, total, error):
                pct = epoch / total * 100
                self.root.after(0, self._update_progress, pct, epoch, total, error)

            errors = self.som.train(
                self.data_info["data"],
                params["epochs"],
                progress_callback=progress_cb,
            )

            self.root.after(0, self._training_done, errors, params)

        except Exception as e:
            self.root.after(0, self._training_error, str(e))

    def _update_progress(self, pct, epoch, total, error):
        """Оновити прогрес-бар (виклик з головного потоку)."""
        self.progress["value"] = pct
        self.status_var.set(f"Навчання: епоха {epoch}/{total}, помилка: {error:.6f}")

    def _training_done(self, errors, params):
        """Навчання завершено."""
        self.is_training = False
        self.btn_train.config(state=tk.NORMAL)
        self.btn_classify.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.NORMAL)
        self.progress["value"] = 100

        final_error = errors[-1] if errors else 0
        accuracy = self.som.compute_accuracy(
            self.data_info["data"],
            self.data_info["labels"],
            self.data_info["n_classes"],
        )

        self.status_var.set(
            f"Навчання завершено. Помилка: {final_error:.6f}, Точність: {accuracy:.2%}"
        )
        self._log(f"Навчання завершено. Помилка: {final_error:.6f}, Точність: {accuracy:.2%}")
        logger.info("Точність класифікації: %.4f", accuracy)

        self._show_visualizations(errors, accuracy)

    def _training_error(self, error_msg):
        """Помилка під час навчання."""
        self.is_training = False
        self.btn_train.config(state=tk.NORMAL)
        self.progress["value"] = 0
        self.status_var.set("Помилка навчання")
        self._log(f"ПОМИЛКА: {error_msg}")
        logger.error("Помилка навчання: %s", error_msg)
        messagebox.showerror("Помилка навчання", error_msg)

    def _show_visualizations(self, errors, accuracy):
        """Відобразити графіки у вкладках."""
        self._close_figures()

        umatrix = self.som.compute_umatrix()
        hit_map = self.som.compute_hit_map(self.data_info["data"])
        label_map = self.som.compute_label_map(
            self.data_info["data"],
            self.data_info["labels"],
            self.data_info["n_classes"],
        )

        fig_u = self.report_gen.plot_umatrix(umatrix, save=False)
        self._embed_figure(fig_u, self.tab_umatrix)

        fig_h = self.report_gen.plot_hit_map(hit_map, save=False)
        self._embed_figure(fig_h, self.tab_hitmap)

        fig_l = self.report_gen.plot_label_map(
            label_map, self.data_info["target_names"], save=False
        )
        self._embed_figure(fig_l, self.tab_labelmap)

        fig_e = self.report_gen.plot_training_error(errors, save=False)
        self._embed_figure(fig_e, self.tab_error)

        self._show_results_text(accuracy, errors)

    def _embed_figure(self, fig, parent):
        """Вбудувати matplotlib figure у вкладку Tkinter."""
        for widget in parent.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()

        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._current_figures.append(fig)

    def _close_figures(self):
        """Закрити попередні figures для звільнення пам'яті."""
        for fig in self._current_figures:
            plt.close(fig)
        self._current_figures.clear()

    def _show_results_text(self, accuracy, errors):
        """Відобразити текстові результати."""
        label_map = self.som.compute_label_map(
            self.data_info["data"],
            self.data_info["labels"],
            self.data_info["n_classes"],
        )
        predictions = self.som.predict(self.data_info["data"], label_map)
        labels = self.data_info["labels"]
        target_names = self.data_info["target_names"]

        lines = [
            "РЕЗУЛЬТАТИ КЛАСИФІКАЦІЇ",
            "=" * 40,
            "",
            f"Загальна точність: {accuracy:.2%}",
            f"Фінальна помилка: {errors[-1]:.6f}" if errors else "",
            "",
            "Точність по класах:",
        ]

        for cls_idx, cls_name in enumerate(target_names):
            mask = labels == cls_idx
            valid = predictions[mask] >= 0
            if valid.sum() > 0:
                cls_acc = np.mean(predictions[mask][valid] == cls_idx)
                lines.append(f"  {cls_name}: {cls_acc:.2%} ({mask.sum()} зразків)")
            else:
                lines.append(f"  {cls_name}: N/A ({mask.sum()} зразків)")

        lines.extend([
            "",
            "Параметри:",
            f"  Карта: {self.current_params['map_rows']}x{self.current_params['map_cols']}",
            f"  Епохи: {self.current_params['epochs']}",
            f"  Швидкість навчання: {self.current_params['learning_rate']}",
            f"  Радіус: {self.current_params['radius']}",
        ])

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.insert(tk.END, "\n".join(lines))
        self.results_text.config(state=tk.DISABLED)

    def _classify(self):
        """Класифікація даних навченою мережею."""
        if self.som is None or not self.som.is_trained:
            messagebox.showwarning("Увага", "Спочатку навчіть мережу.")
            return

        accuracy = self.som.compute_accuracy(
            self.data_info["data"],
            self.data_info["labels"],
            self.data_info["n_classes"],
        )
        self._log(f"Класифікація завершена. Точність: {accuracy:.2%}")
        self.status_var.set(f"Класифікація завершена. Точність: {accuracy:.2%}")

        self._show_results_text(accuracy, self.som.training_errors)
        self.notebook.select(self.tab_results)

    def _save_report(self):
        """Зберегти повний звіт."""
        if self.som is None or not self.som.is_trained:
            messagebox.showwarning("Увага", "Спочатку навчіть мережу.")
            return

        try:
            self.status_var.set("Збереження звіту...")

            umatrix = self.som.compute_umatrix()
            hit_map = self.som.compute_hit_map(self.data_info["data"])
            label_map = self.som.compute_label_map(
                self.data_info["data"],
                self.data_info["labels"],
                self.data_info["n_classes"],
            )

            self.report_gen.plot_umatrix(umatrix, save=True)
            self.report_gen.plot_hit_map(hit_map, save=True)
            self.report_gen.plot_label_map(
                label_map, self.data_info["target_names"], save=True
            )
            self.report_gen.plot_training_error(self.som.training_errors, save=True)
            plt.close("all")

            accuracy = self.som.compute_accuracy(
                self.data_info["data"],
                self.data_info["labels"],
                self.data_info["n_classes"],
            )
            final_error = (
                self.som.training_errors[-1] if self.som.training_errors else 0
            )

            report_text = self.report_gen.generate_text_report(
                self.current_params,
                accuracy,
                final_error,
                self.data_loader.get_summary(),
            )

            output_dir = self.config.get("output_dir", "output")
            self.status_var.set(f"Звіт збережено у каталог {output_dir}/")
            self._log(f"Звіт збережено у каталог {output_dir}/")
            messagebox.showinfo(
                "Збережено",
                f"Звіт та графіки збережено у каталог:\n{os.path.abspath(output_dir)}",
            )
        except Exception as e:
            msg = f"Помилка збереження звіту: {e}"
            self._log(msg)
            logger.error(msg, exc_info=True)
            messagebox.showerror("Помилка", msg)

    def _reset(self):
        """Скинути мережу та результати."""
        if self.is_training:
            messagebox.showwarning("Увага", "Зачекайте завершення навчання.")
            return

        self.som = None
        self._close_figures()
        self.btn_classify.config(state=tk.DISABLED)
        self.btn_report.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_var.set("Готово")

        for tab in [self.tab_umatrix, self.tab_hitmap, self.tab_labelmap, self.tab_error]:
            for widget in tab.winfo_children():
                widget.destroy()
            lbl = ttk.Label(tab, text="Натисніть «Навчити мережу» для початку роботи",
                            font=("Segoe UI", 12))
            lbl.pack(expand=True)

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        self.results_text.config(state=tk.DISABLED)

        params = self.config.get_default_params()
        for key, val in params.items():
            if key in self.param_vars:
                self.param_vars[key].set(str(val))

        self._log("Мережу скинуто. Параметри відновлено за замовчуванням.")
        logger.info("Скидання мережі")

    def _log(self, message: str):
        """Додати повідомлення у журнал подій GUI."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def _show_about(self):
        """Діалог «Про програму»."""
        version = self.config.get("version", "1.0.0")
        messagebox.showinfo(
            "Про програму",
            f"Класифікація мережею Кохонена (SOM)\n"
            f"Версія: {version}\n\n"
            f"Програма для класифікації об'єктів за допомогою\n"
            f"самоорганізаційної карти Кохонена.\n\n"
            f"Датасет: Iris\n"
            f"Мова: Python + Tkinter + NumPy + Matplotlib",
        )
