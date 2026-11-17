"""
gui.py - Tkinter desktop interface.

Layout: a navy sidebar with four buttons (Dashboard, Model Training,
Visualizations, Prediction) and a content area that shows one page at a time.
"""

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

import visualizations as viz
from model import INPUT_FEATURES, MAX_DEPTH

# Consistent colour scheme
NAVY = "#1f2d3d"
ACCENT = "#2980b9"
BG = "#f4f6f8"
CARD = "#ffffff"
TEXT = "#2c3e50"
RED = "#c0392b"
GREEN = "#27ae60"

DISCLAIMER = ("Educational project only - predictions are NOT a medical "
              "diagnosis. Consult a qualified doctor for real cases.")


class App(tk.Tk):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.title("Breast Cancer Classification using CART")
        self.geometry("1100x720")
        self.minsize(950, 650)
        self.configure(bg=BG)
        self._setup_style()

        # Sidebar
        sidebar = tk.Frame(self, bg=NAVY, width=200)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="CART\nClassifier", bg=NAVY, fg="white",
                 font=("Segoe UI", 16, "bold"), pady=25).pack()

        # Content area + disclaimer footer
        right = tk.Frame(self, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        tk.Label(right, text=DISCLAIMER, bg="#fdecea", fg=RED,
                 font=("Segoe UI", 9, "italic"), pady=4).pack(side="bottom",
                                                              fill="x")
        self.content = tk.Frame(right, bg=BG)
        self.content.pack(fill="both", expand=True, padx=20, pady=15)

        # Build pages
        self.pages = {}
        self.nav_buttons = {}
        builders = [("Dashboard", self._build_dashboard),
                    ("Model Training", self._build_training),
                    ("Visualizations", self._build_visuals),
                    ("Prediction", self._build_prediction)]
        for name, builder in builders:
            frame = tk.Frame(self.content, bg=BG)
            self.pages[name] = frame
            builder(frame)
            btn = tk.Button(sidebar, text=name, anchor="w", padx=20,
                            bg=NAVY, fg="white", bd=0, relief="flat",
                            activebackground=ACCENT, activeforeground="white",
                            font=("Segoe UI", 11), cursor="hand2",
                            command=lambda n=name: self.show_page(n))
            btn.pack(fill="x", ipady=8)
            self.nav_buttons[name] = btn
        self.show_page("Dashboard")

    # ----------------------------------------------------------------- utils
    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Accent.TButton", background=ACCENT, foreground="white",
                        font=("Segoe UI", 11, "bold"), padding=8)
        style.map("Accent.TButton", background=[("active", "#21618c")])
        style.configure("Treeview", font=("Segoe UI", 10), rowheight=24)
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"))

    def show_page(self, name):
        for n, frame in self.pages.items():
            frame.pack_forget()
            self.nav_buttons[n].configure(bg=NAVY)
        self.pages[name].pack(fill="both", expand=True)
        self.nav_buttons[name].configure(bg=ACCENT)

    def _heading(self, parent, text):
        tk.Label(parent, text=text, bg=BG, fg=TEXT,
                 font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(0, 10))

    def _card(self, parent, title, value, color=ACCENT):
        """A small statistic card used on the Dashboard and Training pages."""
        card = tk.Frame(parent, bg=CARD, highlightbackground="#dfe4ea",
                        highlightthickness=1)
        tk.Label(card, text=title, bg=CARD, fg="#7f8c8d",
                 font=("Segoe UI", 10)).pack(padx=20, pady=(12, 0))
        lbl = tk.Label(card, text=value, bg=CARD, fg=color,
                       font=("Segoe UI", 22, "bold"))
        lbl.pack(padx=20, pady=(0, 12))
        return card, lbl

    # ------------------------------------------------------------- dashboard
    def _build_dashboard(self, page):
        self._heading(page, "Dashboard")
        info = self.model.dataset_info()

        tk.Label(page, bg=BG, fg=TEXT, justify="left", wraplength=800,
                 font=("Segoe UI", 11),
                 text=("This project classifies breast tumour samples as "
                       "Benign (non-cancerous) or Malignant (cancerous) using "
                       "the CART (Classification and Regression Trees) "
                       "algorithm - scikit-learn's DecisionTreeClassifier "
                       "with the Gini impurity criterion. Data: Breast Cancer "
                       "Wisconsin (Diagnostic) dataset.")
                 ).pack(anchor="w", pady=(0, 15))

        row = tk.Frame(page, bg=BG)
        row.pack(anchor="w", pady=5)
        for title, val in [("Total Records", info["records"]),
                           ("Features", info["features"]),
                           ("Classes", info["classes"]),
                           ("Benign", info["benign"]),
                           ("Malignant", info["malignant"])]:
            card, _ = self._card(row, title, str(val))
            card.pack(side="left", padx=(0, 12))

        tk.Label(page, text="Sample records (first 5 rows, target: 0 = "
                 "malignant, 1 = benign)", bg=BG, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(20, 5))
        sample = self.model.sample_records()
        cols = list(sample.columns)
        tv = ttk.Treeview(page, columns=cols, show="headings", height=5)
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, width=130, anchor="center")
        for _, r in sample.iterrows():
            tv.insert("", "end", values=[round(v, 3) for v in r.values])
        tv.pack(anchor="w")

        tk.Label(page, bg=BG, fg=TEXT, justify="left", wraplength=800,
                 font=("Segoe UI", 10),
                 text=("Benign tumours do not invade nearby tissue or spread. "
                       "Malignant tumours are cancerous and can invade other "
                       "tissue. Missing values in dataset: "
                       f"{info['missing']}.")
                 ).pack(anchor="w", pady=(15, 0))

    # -------------------------------------------------------------- training
    def _build_training(self, page):
        self._heading(page, "Model Training")
        tk.Label(page, bg=BG, fg=TEXT, font=("Segoe UI", 11), justify="left",
                 text=(f"Algorithm: CART (DecisionTreeClassifier)   |   "
                       f"Criterion: Gini   |   Max depth: {MAX_DEPTH}   |   "
                       "Split: 80% train / 20% test")
                 ).pack(anchor="w", pady=(0, 12))

        ttk.Button(page, text="Train CART Model", style="Accent.TButton",
                   command=self.train_model).pack(anchor="w")
        self.status_var = tk.StringVar(value="Status: model not trained yet.")
        tk.Label(page, textvariable=self.status_var, bg=BG, fg=TEXT,
                 font=("Segoe UI", 11, "italic")).pack(anchor="w", pady=10)

        self.metric_row = tk.Frame(page, bg=BG)
        self.metric_row.pack(anchor="w", pady=5)
        self.metric_labels = {}
        for name in ["Accuracy", "Precision", "Recall", "F1-score"]:
            card, lbl = self._card(self.metric_row, name, "--")
            card.pack(side="left", padx=(0, 12))
            self.metric_labels[name] = lbl

        self.note_var = tk.StringVar()
        tk.Label(page, textvariable=self.note_var, bg=BG, fg=TEXT,
                 font=("Segoe UI", 10), justify="left", wraplength=800
                 ).pack(anchor="w", pady=15)

    def train_model(self):
        try:
            self.config(cursor="watch")
            self.update_idletasks()
            m = self.model.train()
        except Exception as exc:
            messagebox.showerror("Training failed", str(exc))
            self.status_var.set("Status: training failed.")
            return
        finally:
            self.config(cursor="")

        self.status_var.set("Status: training completed successfully.")
        for name, lbl in self.metric_labels.items():
            lbl.configure(text=f"{m[name] * 100:.2f}%")
        self.note_var.set(
            f"Training accuracy: {m['Train accuracy'] * 100:.2f}%  (compare "
            "with test accuracy to check overfitting).\nPrecision, recall and "
            "F1 are computed treating Malignant as the positive class.")
        self._refresh_visuals()

    # ---------------------------------------------------------- visualizations
    def _build_visuals(self, page):
        self._heading(page, "Visualizations")
        self.tabs = ttk.Notebook(page)
        self.tabs.pack(fill="both", expand=True)
        self.tab_frames = {}
        for name in ["Class Distribution", "Confusion Matrix", "Decision Tree"]:
            f = tk.Frame(self.tabs, bg=CARD)
            self.tabs.add(f, text=name)
            self.tab_frames[name] = f
        # Class distribution does not need a trained model
        self._embed(self.tab_frames["Class Distribution"],
                    viz.class_distribution_figure(self.model.df,
                                                  self.model.class_names))
        for name in ["Confusion Matrix", "Decision Tree"]:
            self._placeholder(self.tab_frames[name])

    def _placeholder(self, frame):
        for w in frame.winfo_children():
            w.destroy()
        tk.Label(frame, text="Train the model first (Model Training page).",
                 bg=CARD, fg="#7f8c8d", font=("Segoe UI", 12)).pack(pady=40)

    def _embed(self, frame, fig):
        """Put a Matplotlib figure (with zoom/save toolbar) inside a frame."""
        for w in frame.winfo_children():
            w.destroy()
        canvas = FigureCanvasTkAgg(fig, master=frame)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        canvas.draw()

    def _refresh_visuals(self):
        m = self.model
        self._embed(self.tab_frames["Confusion Matrix"],
                    viz.confusion_matrix_figure(m.cm, m.class_names))
        self._embed(self.tab_frames["Decision Tree"],
                    viz.decision_tree_figure(m.clf, m.feature_names,
                                             m.class_names))

    # ------------------------------------------------------------ prediction
    def _build_prediction(self, page):
        self._heading(page, "Prediction")
        tk.Label(page, bg=BG, fg=TEXT, font=("Segoe UI", 10), justify="left",
                 text=("Enter tumour measurements (the other 22 features are "
                       "estimated from them by linear regression). Hint shows the "
                       "min - max range in the dataset.")
                 ).pack(anchor="w", pady=(0, 10))

        form = tk.Frame(page, bg=CARD, highlightbackground="#dfe4ea",
                        highlightthickness=1)
        form.pack(anchor="w", padx=(0, 10), ipadx=15, ipady=10)
        ranges = self.model.feature_ranges()
        self.entries = {}
        for i, f in enumerate(INPUT_FEATURES):
            lo, hi, _ = ranges[f]
            tk.Label(form, text=f.title(), bg=CARD, fg=TEXT,
                     font=("Segoe UI", 10)).grid(row=i, column=0, sticky="w",
                                                 padx=15, pady=4)
            e = ttk.Entry(form, width=14)
            e.grid(row=i, column=1, padx=10)
            tk.Label(form, text=f"({lo:.3g} - {hi:.3g})", bg=CARD,
                     fg="#7f8c8d", font=("Segoe UI", 9)).grid(row=i, column=2,
                                                               sticky="w")
            self.entries[f] = e

        btns = tk.Frame(page, bg=BG)
        btns.pack(anchor="w", pady=10)
        ttk.Button(btns, text="Predict", style="Accent.TButton",
                   command=self.predict).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Load typical benign",
                   command=lambda: self._fill(1)).pack(side="left", padx=4)
        ttk.Button(btns, text="Load typical malignant",
                   command=lambda: self._fill(0)).pack(side="left", padx=4)
        ttk.Button(btns, text="Clear",
                   command=self._clear).pack(side="left", padx=4)

        self.result_var = tk.StringVar(value="")
        self.result_lbl = tk.Label(page, textvariable=self.result_var, bg=BG,
                                   font=("Segoe UI", 18, "bold"))
        self.result_lbl.pack(anchor="w", pady=5)

    def _fill(self, label):
        vals = self.model.class_example(label)
        for f, e in self.entries.items():
            e.delete(0, "end")
            e.insert(0, f"{vals[f]:.4g}")

    def _clear(self):
        for e in self.entries.values():
            e.delete(0, "end")
        self.result_var.set("")

    def predict(self):
        if not self.model.is_trained:
            messagebox.showwarning("Model not trained",
                                   "Please train the model first.")
            self.show_page("Model Training")
            return
        values = {}
        for f, e in self.entries.items():
            txt = e.get().strip()
            try:
                v = float(txt)
            except ValueError:
                messagebox.showerror("Invalid input",
                                     f"'{f}' must be a number (got '{txt}').")
                return
            if v <= 0:
                messagebox.showerror("Invalid input",
                                     f"'{f}' must be greater than 0.")
                return
            values[f] = v
        try:
            label, prob = self.model.predict(values)
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return
        color = GREEN if label == "benign" else RED
        self.result_lbl.configure(fg=color)
        self.result_var.set(f"Prediction: {label.upper()}  "
                            f"(confidence {prob * 100:.1f}%)")
