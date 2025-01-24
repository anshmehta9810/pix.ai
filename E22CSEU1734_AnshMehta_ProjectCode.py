import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageOps
import customtkinter as ctk
import pytesseract
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ASUS\tesseract.exe"

class ImageApp:
    def _init_(self, root):
        self.root = root
        self.root.geometry("800x800")
        self.root.title("Image Transformation App")
        ctk.set_appearance_mode("dark")

        # Main menu
        self.menu_frame = ctk.CTkFrame(self.root)
        self.menu_frame.pack(fill="both", expand=True, padx=150, pady=150)

        self.title_label = ctk.CTkLabel(self.menu_frame, text="Image Transformation Menu", font=("Arial", 24))
        self.title_label.pack(pady=10)

        self.text_to_image_button = ctk.CTkButton(self.menu_frame, text="Text-to-Image Generator", command=self.open_text_to_image)
        self.text_to_image_button.pack(pady=10)

        self.image_to_text_button = ctk.CTkButton(self.menu_frame, text="Image-to-Text Converter", command=self.open_image_to_text)
        self.image_to_text_button.pack(pady=10)

        self.image_resizer_button = ctk.CTkButton(self.menu_frame, text="Image Resizer", command=self.open_image_resizer)
        self.image_resizer_button.pack(pady=10)

        self.image_filter_button = ctk.CTkButton(self.menu_frame, text="Image Filter", command=self.open_image_filter)
        self.image_filter_button.pack(pady=10)

    def open_text_to_image(self):
        self.menu_frame.pack_forget()
        TextToImage(self.root, self.menu_frame)

    def open_image_to_text(self):
        self.menu_frame.pack_forget()
        ImageToText(self.root, self.menu_frame)

    def open_image_resizer(self):
        self.menu_frame.pack_forget()
        ImageResizer(self.root, self.menu_frame)

    def open_image_filter(self):
        self.menu_frame.pack_forget()
        ImageFilter(self.root, self.menu_frame)         


class TextToImage:
    def _init_(self, root, parent_frame):
        self.root = root
        self.parent_frame = parent_frame

        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(fill="both", expand=True, padx=20, pady=20)

        self.prompt = ctk.CTkEntry(master=self.frame, height=40, width=512, font=("Arial", 20), text_color="black", fg_color="white")
        self.prompt.pack(side="top", fill="x", pady=(0, 10))

        self.lmain = ctk.CTkLabel(master=self.frame, height=512, width=512)
        self.lmain.pack(side="top", pady=(0, 10))

        self.trigger = ctk.CTkButton(master=self.frame, height=40, width=120, font=("Arial", 20), text="Generate", command=self.generate)
        self.trigger.pack(side="top", pady=(0, 10))

        self.back_button = ctk.CTkButton(master=self.frame, text="Back", command=self.go_back)
        self.back_button.pack(side="top", pady=(0, 10))
        

        # Model initialization
        modelid = "CompVis/stable-diffusion-v1-4"
        device = "cuda"
        self.pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float32, use_auth_token=auth_token)
        self.pipe.to(device)

    def generate(self):

        device="cuda"
        with autocast(device):
            output = self.pipe(self.prompt.get(), guidance_scale=8.5)
        

        if 'images' in output:
            image = output['images'][0]
            image.save('generatedimage.png')
        
            img = ImageTk.PhotoImage(image)
            self.lmain.configure(image=img)
            self.lmain.image = img
            

    def go_back(self):
        self.frame.pack_forget()
        self.parent_frame.pack()


class ImageToText:
    def _init_(self, root, parent_frame):
        self.root = root
        self.parent_frame = parent_frame

        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(fill="both", expand=True)

        self.upload_button = ctk.CTkButton(self.frame, text="Upload Image", command=self.upload_image)


        self.upload_button.pack(pady=20)

        self.result_label = ctk.CTkLabel(self.frame, text="", wraplength=500)
        self.result_label.pack(pady=10)

        self.back_button = ctk.CTkButton(self.frame, text="Back", command=self.go_back)
        self.back_button.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
        if file_path:
            text = pytesseract.image_to_string(Image.open(file_path))
            self.result_label.configure(text=text)

    def go_back(self):
        self.frame.pack_forget()
        self.parent_frame.pack()


class ImageResizer:
    def _init_(self, root, parent_frame):
        self.root = root
        self.parent_frame = parent_frame

        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(fill="both", expand=True)

        self.upload_button = ctk.CTkButton(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Resize inputs
        self.width_label = ctk.CTkLabel(self.frame, text="Width:")
        self.width_label.pack(pady=5)
        self.width_entry = ctk.CTkEntry(self.frame, width=100)
        self.width_entry.pack(pady=5)

        self.height_label = ctk.CTkLabel(self.frame, text="Height:")
        self.height_label.pack(pady=5)
        self.height_entry = ctk.CTkEntry(self.frame, width=100)
        self.height_entry.pack(pady=5)


        

    def upload_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
        if self.file_path:
            self.image = Image.open(self.file_path)

    def apply_changes(self):
        if hasattr(self, "image"):
            # Apply resizing
            width = int(self.width_entry.get()) if self.width_entry.get() else self.image.width
            height = int(self.height_entry.get()) if self.height_entry.get() else self.image.height
            resized_image = self.image.resize((width, height))


            # Save and display
            resized_image.save("processed_image.png")
            print(f"Image saved as processed_image.png.")

    def go_back(self):
        self.frame.pack_forget()
        self.parent_frame.pack()

class ImageFilter:
    def _init_(self, root, parent_frame):
        self.root = root
        self.parent_frame = parent_frame

        self.frame = ctk.CTkFrame(self.root)
        self.frame.pack(fill="both", expand=True)

        self.upload_button = ctk.CTkButton(self.frame, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Filter dropdown
        self.filter_label = ctk.CTkLabel(self.frame, text="Select Filter:")
        self.filter_label.pack(pady=5)
        self.filters = ["None", "Grayscale", "Binary", "Negative"]
        self.selected_filter = tk.StringVar(value=self.filters[0])
        self.filter_dropdown = ttk.Combobox(self.frame, textvariable=self.selected_filter, values=self.filters)
        self.filter_dropdown.pack(pady=5)

        self.process_button = ctk.CTkButton(self.frame, text="Apply", command=self.apply_changes)
        self.process_button.pack(pady=10)

        self.back_button = ctk.CTkButton(self.frame, text="Back", command=self.go_back)
        self.back_button.pack(pady=10)

    def upload_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
        if self.file_path:
            self.image = Image.open(self.file_path)

    def apply_changes(self):
        if hasattr(self, "image"):
            filtered_image = self.image
            # Apply filter
            filter_choice = self.selected_filter.get()
            if filter_choice == "Grayscale":
                filtered_image= ImageOps.grayscale(filtered_image)
            elif filter_choice == "Binary":
                filtered_image = filtered_image.convert("L").point(lambda x: 0 if x < 128 else 255, "1")
            elif filter_choice == "Negative":
                filtered_image = ImageOps.invert(filtered_image.convert("RGB"))

            # Save and display
            filtered_image.save("processed_image.png")
            print(f"Image saved as processed_image.png with {filter_choice} filter applied.")

    def go_back(self):
        self.frame.pack_forget()
        self.parent_frame.pack()



# Run the app
if _name_ == "_main_":
    root = ctk.CTk()
    app = ImageApp(root)
    root.geometry("800x700")  
    root.mainloop()