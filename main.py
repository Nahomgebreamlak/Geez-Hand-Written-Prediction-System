import tkinter as tk
from tkinter import *
from tkinter import messagebox
from Digit_class import DigitRecognize as DG

window = tk.Tk()
Dclass = DG()
window.title(" ናይ ትግርኛ ፊደላት Hand written Geez alphabet recognition")

bg = PhotoImage(file="alphabets.png")
l1 = tk.Label(window, text="Alphabet", font=('Arial', 20))

l1.place(x=5, y=0)

t1 = tk.Entry(window, width=20, border=5)
t1.place(x=150, y=0)

b1 = tk.Button(window, text="1. Open paint and capture the screen", font=("Arial", 15), bg="black", fg="white", borderwidth=0,


               command=lambda: Dclass.collect_image(alphabet=t1.get()))
b1.place(x=5, y=50)
b2 = tk.Button(window, text ="2. Generate Dataset ",font=("Arial", 15), bg="black", fg="yellow", command=lambda: Dclass.generate_dataset())
b2.place(x=5, y=100)

b3 = tk.Button(window, text="3.Train the model and calculate accuracy", font=("Arial", 15), bg="black",
               fg="white", command=lambda: Dclass.train_and_calculate_accuracy())
b3.place(x=5, y=150)


b4 = tk.Button(window, text="4. Live Prediction", font=("Arial", 15), bg="black", fg="yellow", command=lambda: Dclass.live_prediction()
               )
b4.place(x=5, y=200)

window.geometry("600x300")
window.mainloop()
