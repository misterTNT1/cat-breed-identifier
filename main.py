import os
import random
import shutil
from tkinter import filedialog, ttk
from tkinter.filedialog import askopenfilename
import threading
import queue

import customtkinter as ctk
import torch
from PIL import Image

import model_handler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cat_model = model_handler.load_model(device)
cat_model.eval()
class_names = model_handler.get_class_names()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class CatBreedQuizApp(ctk.CTk):
    PATH = "training_new/test"
    def __init__(self):
        super().__init__()
        self.title("Cat Breed Quiz")
        self.geometry("800x700")
        self.configure(fg_color="#46178f")

        self.selected_breed = ctk.StringVar(value="")
        self.current_image_path = None
        self.correct_answer = None
        self.total_images = 0
        self.score = 0
        self.started = False
        self.queue = queue.Queue()
        self.running = False

        self.create_widgets()

    def create_widgets(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=30, pady=20)

        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text="Guess the Cat Breed!",
            font=ctk.CTkFont(size=32, weight="bold"),
            text_color="white"
        )
        self.title_label.pack(pady=(0, 20))

        self.image_label = ctk.CTkLabel(
            self.main_frame,
            font=ctk.CTkFont(size=18),
            text="",
            text_color="#888888"
        )
        self.image_label.pack(pady=20)

        self.answers_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.answers_frame.pack(pady=25, fill="x")

        self.answers_frame.grid_columnconfigure((0, 1), weight=1)

        self.answer_buttons = []

        self.result_label = ctk.CTkLabel(
            self.main_frame,
            text="Select your answer above!",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white"
        )
        self.result_label.pack(pady=15)

        self.upload_button = ctk.CTkButton(
            self.main_frame,
            text="upload your folder with cats",
            command=self.add_images
        )

        self.upload_button.pack(pady=10)

        self.start_btn = ctk.CTkButton(
            self.main_frame,
            text="Start",
            command=self.show_random_image

        )
        self.start_btn.pack(pady=10)

        self.submit_btn = ctk.CTkButton(
            self.main_frame,
            text="Submit Guess",
            command=self.submit_guess
        )
        self.submit_btn.pack(pady=10)

        self.score_label = ctk.CTkLabel(
            self.main_frame,
            text=f"Score: {self.score}/{self.total_images}",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white"
        )

        self.score_label.pack(pady=(10, 20))

        self.progress = ttk.Progressbar(self.main_frame, length=300, mode="determinate")

    # ----------------- IMAGE -----------------
    def show_random_image(self, path=PATH):
        self.start_btn.pack_forget()
        self.upload_button.pack_forget()
        images = []
        image_folders = os.listdir(path)
        for folder in image_folders:
            folder_path = os.path.join(path, folder)
            images.extend([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

        if os.path.exists("user_images"):
            user_images = [
                os.path.join("user_images", f)
                for f in os.listdir("user_images")
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            images.extend(user_images)
        if not images:
            return
        self.current_image_path = random.choice(images)
        img = Image.open(self.current_image_path)
        img.thumbnail((400, 400))

        ctk_image = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=(img.width, img.height)
        )

        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image

        self.generate_options()


    def process_images(self, folder_path, files):
        os.makedirs("user_images", exist_ok=True)

        for i, file in enumerate(files):
            src = os.path.join(folder_path, file)
            dst = os.path.join("user_images", file)

            name, ext = os.path.splitext(file)
            counter = 1
            while os.path.exists(dst):
                dst = os.path.join("user_images", f"{name}_{counter}{ext}")
                counter += 1

            shutil.copy2(src, dst)

            # send progress update
            self.queue.put(i+1)

        self.queue.put("DONE")

    def check_queue(self):
        try:
            msg = self.queue.get_nowait()

            if msg == "DONE":
                self.progress["value"] = self.progress["maximum"]
                self.running = False
                self.progress.pack_forget()
                return
            else:
                self.progress["value"] = msg

        except queue.Empty:
            pass

        self.main_frame.after(100, self.check_queue)

    def add_images(self):
        if self.running:
            return
        os.makedirs("user_images", exist_ok=True)
        folder_path = filedialog.askdirectory()

        if not folder_path:
            return

        files = [
            f for f in os.listdir(folder_path)
            if os.path.isfile(os.path.join(folder_path, f))
            and f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not files:
            return

        self.progress.pack(pady=10)
        self.progress["value"] = 0
        self.progress["maximum"] = len(files)

        self.running = True

        thread = threading.Thread(
            target=self.process_images,
            args=(folder_path, files),
            daemon=True
        )

        thread.start()

        self.main_frame.after(100, self.check_queue)


    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        self.current_image_path = file_path

        img = Image.open(file_path)
        img.thumbnail((400, 400))

        ctk_image = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=(img.width, img.height)
        )

        self.image_label.configure(image=ctk_image, text="")
        self.image_label.image = ctk_image

        self.generate_options()

    # ----------------- QUIZ LOGIC -----------------

    def generate_options(self):
        for btn in self.answer_buttons:
            btn.destroy()

        self.answer_buttons.clear()

        self.correct_answer = model_handler.predict_cat_breed(
            cat_model, self.current_image_path, device
        )

        wrong_answers = [c for c in class_names if c != self.correct_answer]
        options = random.sample(wrong_answers, 3)
        options.append(self.correct_answer)
        random.shuffle(options)

        for i, option in enumerate(options):
            btn = ctk.CTkButton(
                self.answers_frame,
                text=option,
                command=lambda o=option: self.select_breed(o)
            )
            btn.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky="ew")
            self.answer_buttons.append(btn)

        self.clear_result()



    def clear_result(self):
        self.selected_breed.set("")
        self.result_label.configure(text="Select your answer above!", text_color="white")
        self.image_label.configure(text="")


    def select_breed(self, breed):
        self.selected_breed.set(breed)
        self.result_label.configure(
            text=f"Selected: {breed}",
            text_color="#ffd700"
        )

    def submit_guess(self):
        if not self.selected_breed.get():
            self.result_label.configure(
                text="Please select a breed first!",
                text_color="orange"
            )
            return

        if self.selected_breed.get() == self.correct_answer:
            self.result_label.configure(
                text="Correct!",
                text_color="#00ff88"
            )

            self.score += 1
        else:
            self.result_label.configure(
                text=f"Wrong! Correct: {self.correct_answer}",
                text_color="#ff4444"
            )

        self.total_images += 1

        self.score_label.configure(
            text=f"Score: {self.score}/{self.total_images}",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="white"
        )

        self.after(750, self.show_random_image)


if __name__ == "__main__":
    app = CatBreedQuizApp()
    app.mainloop()