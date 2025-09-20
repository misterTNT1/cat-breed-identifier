import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageFile

root = tk.Tk()
root.title("Cat Breed Classifier")
root.geometry("650x650")
style = ttk.Style(root)
style.theme_use("clam")

guess = tk.StringVar()
answer = tk.StringVar(value="Your guess will appear here")

# Image frame
image_frame = tk.Frame(root, bd=2, relief="groove", bg="#f0f0f0")
image_frame.grid(row=0, column=0, columnspan=5, pady=10)
label = tk.Label(image_frame, bg="#f0f0f0")
label.pack(padx=10, pady=10)

# Options
options = ["himalayan cat", "domestic shorthair cat", "siberian cat", "American bobtail"]
radio_frame = tk.LabelFrame(root, text="Choose a Breed", padx=10, pady=10)
radio_frame.grid(row=1, column=0, columnspan=5, pady=10)

for option in options:
    ttk.Radiobutton(radio_frame, text=option, variable=guess, value=option).pack(anchor="w", pady=2)
guess.set(options[0])

# Result
result = tk.Label(root, textvariable=answer, font=("Helvetica", 14, "bold"))
result.grid(row=2, column=0, columnspan=5, pady=10)

def resize_image(image: ImageFile, new_size):
    return image.resize(size=new_size)

def load_image(file_path):
    try:
        image_content = Image.open(file_path)
        image_content = resize_image(image_content, (400, 400))
        global tk_image
        tk_image = ImageTk.PhotoImage(image_content)
        label.configure(image=tk_image)
    except Exception as e:
        print("error handling file ", e)

default_image = "cat.jpg"
load_image(default_image)

def upload_image():
    file_path = filedialog.askopenfilename()
    load_image(file_path if file_path else default_image)

def submit():
    answer.set(f"You chose: {guess.get()}")

# Buttons
ttk.Button(root, text="📁 Upload Image", command=upload_image).grid(row=3, column=2, pady=5)
ttk.Button(root, text="✔ Submit Guess", command=submit).grid(row=4, column=2, pady=5)

for i in range(5):
    root.grid_columnconfigure(i, weight=1)

root.mainloop()
