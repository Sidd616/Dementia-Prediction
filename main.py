"""Entry point: load a cached trained model (training and caching one if
none exists yet), then launch the GUI.
"""
import customtkinter as ctk

from gui import DementiaPredictionGUI
from model import load_or_train_model


def main():
    model = load_or_train_model()

    root = ctk.CTk()
    DementiaPredictionGUI(root, model=model)
    root.mainloop()


if __name__ == "__main__":
    main()
