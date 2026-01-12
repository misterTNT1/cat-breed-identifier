import random
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter import messagebox
import torch
from PIL import Image, ImageTk, ImageFile

import model_handler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cat_model = model_handler.load_model(device)

cat_model.eval()

root = tk.Tk()
root.configure(bg="#f0f0f0")
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
options = model_handler.get_class_names()
radio_frame = tk.LabelFrame(root, padx=10, pady=10)
radio_frame.grid(row=1, column=0, columnspan=5, pady=10)
radio_frame.rowconfigure(0, weight=1)
radio_frame.rowconfigure(1, weight=1)
radio_frame.columnconfigure(0, weight=1)
radio_frame.columnconfigure(1, weight=1)


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


def load_options():
    # removing the old options
    for widget in radio_frame.winfo_children():
        widget.destroy()

    estimated = model_handler.predict_cat_breed(cat_model, current_image_path, device)
    remaining_values = [value for value in options if value != estimated]
    option_list = random.sample(remaining_values, 3)
    option_list.append(estimated)
    colors = ["#FF3030", "#5BC0EB", "#FFF78A", "#41D95D"]
    for i, option in enumerate(zip(option_list, colors)):
        row = i // 2
        col = i % 2
        tk.Radiobutton(radio_frame, text=option[0], variable=guess, value=option[0], background=option[1], width=0, height=0).grid(
            row=row, column=col, padx=5, pady=5, sticky="nsew"
        )

    guess.set(options[0])

default_image = "test_siamese.jpg"
load_image(default_image)
current_image_path = default_image
load_options()


def upload_image():
    file_path = filedialog.askopenfilename()
    if not file_path.endswith(".jpg"):
        messagebox.showerror("wrong file path", f"this file path does not contain the proper file path"
                                                f"\nfile path: {file_path}")
        return
    global current_image_path
    current_image_path = file_path
    load_image(file_path if file_path else default_image)
    load_options()

# prints the model's guess as well as your guess
def submit():
    estimated_breed = model_handler.predict_cat_breed(cat_model, current_image_path, device)
    answer.set(f"You chose: {guess.get()}, model's guess: {estimated_breed}")
# Buttons
ttk.Button(root, text="Upload Image", command=upload_image).grid(row=3, column=2, pady=5)
ttk.Button(root, text="Submit Guess", command=submit).grid(row=4, column=2, pady=5)

for i in range(5):
    root.grid_columnconfigure(i, weight=1)

root.mainloop()
