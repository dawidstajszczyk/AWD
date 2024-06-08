import tkinter as tk
from tkinter import ttk
from CF import CF
from CBF import CBF
from hybrid import hybrid_recommendation

title = 'Waiting to Exhale (1995)'

def show_CF_recommendation():
    recommendation, _ = CF(title)
    label.config(text=f"Collaborative Filtering Recommendations\n \nNa podstawie filmu: {title}\n{recommendation}")

def show_CBF_recommendation():
    recommendation, _ = CBF(title)
    label.config(text=f"Content Based Filtering Recommendations\n \nNa podstawie filmu: {title}\n{recommendation}")

def show_hybrid_recommendation():
    recommendation = hybrid_recommendation(title)
    label.config(text=f"Hybrid Recommendations\n \nNa podstawie filmu: {title}\n{recommendation}")

# Utwórz główne okno aplikacji
root = tk.Tk()
root.title("Podstawowe GUI")

# Dodaj etykietę do okna
label = ttk.Label(root, text="Witaj w GUI!")
label.pack(padx=100, pady=50)

# Dodaj przycisk do okna
button = ttk.Button(root, text="Pokaż rekomendację CF", command=show_CF_recommendation)
button.pack(padx=10, pady=10)

# Dodaj przycisk do okna
button = ttk.Button(root, text="Pokaż rekomendację CBF", command=show_CBF_recommendation)
button.pack(padx=10, pady=10)

# Dodaj przycisk do okna
button = ttk.Button(root, text="Pokaż rekomendację hybrydową", command=show_hybrid_recommendation)
button.pack(padx=10, pady=10)

# Uruchom główną pętlę zdarzeń
root.mainloop()
