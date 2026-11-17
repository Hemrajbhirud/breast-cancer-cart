"""
main.py - Entry point. Run with:  python main.py
"""

import sys
from tkinter import messagebox

from model import CARTModel
from gui import App


def main():
    try:
        model = CARTModel()      # loads the dataset
        app = App(model)         # builds the window
        app.mainloop()
    except Exception as exc:
        messagebox.showerror("Error", f"Application failed to start:\n{exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
