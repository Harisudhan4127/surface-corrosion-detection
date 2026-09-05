import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageTk
import yaml

# Add src to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data.transforms import get_val_transform
from models.model import build_model
from utils.logger import setup_logger

# Load config
with open(ROOT / "configs" / "config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

CLASS_NAMES = cfg["data"]["class_names"]

# Colour per class for overlay titles
CLASS_COLORS = {
    "Crazing": "#e74c3c",
    "Inclusion": "#e67e22",
    "Patches": "#f1c40f",
    "Pitted_surface": "#2ecc71",
    "Rolled-in_scale": "#3498db",
    "Scratches": "#9b59b6",
}


class CorrosionDetector:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_model(cfg)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=False))
        self.model.to(self.device)
        self.model.eval()
        self.transform = get_val_transform()

    def predict_image(self, img_path: str) -> tuple[str, float, np.ndarray]:
        image = Image.open(img_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        return CLASS_NAMES[pred_idx], confidence, np.array(image)

    def predict_image(self, img_path: str) -> tuple[str, float, np.ndarray]:
        image = Image.open(img_path).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        return CLASS_NAMES[pred_idx], confidence, np.array(image)


class CorrosionDetectionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Surface Corrosion Detection')
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        self.geometry('1180x780')
        self.resizable(False, False)

        # Try to load model
        checkpoint_path = ROOT / "models" / "checkpoints" / "best_model.pth"
        if not checkpoint_path.exists():
            messagebox.showerror("Error", f"Model checkpoint not found: {checkpoint_path}")
            self.destroy()
            return

        try:
            self.detector = CorrosionDetector(str(checkpoint_path))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.destroy()
            return

        self.image_path = None
        self.photo_image = None

        self.create_widgets()

    def create_widgets(self):
        control_frame = tk.Frame(self, padx=12, pady=8)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text='Open Image', command=self.open_image, width=15).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text='Predict', command=self.predict_image, width=15).pack(side=tk.LEFT, padx=4)

        self.status_label = tk.Label(self, text='Ready to predict', font=('Arial', 16), bg='#202020', fg='white', padx=12, pady=8)
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Label(self, bg='black')
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.info_label = tk.Label(self, text='Load an image and click Predict to classify surface defects.', font=('Arial', 12), fg='#f0f0f0', bg='#101010', anchor='w')
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def open_image(self):
        path = filedialog.askopenfilename(
            title='Select image',
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp')],
        )
        if not path:
            return

        self.image_path = path
        image = Image.open(path)
        self.show_image(image)
        self.status_label.config(text='Image loaded: ' + os.path.basename(path), bg='#1f3b6f')

    def predict_image(self):
        if not self.image_path:
            messagebox.showwarning('Predict', 'Please load an image first.')
            return

        try:
            pred_class, confidence, img_array = self.detector.predict_image(self.image_path)
            self.show_image(Image.fromarray(img_array))
            color = CLASS_COLORS.get(pred_class, "#ffffff")
            self.status_label.config(
                text=f'Prediction: {pred_class} ({confidence:.1%})',
                bg=color,
                fg='white' if color != "#f1c40f" else 'black'
            )
            self.info_label.config(text=f'Defect type: {pred_class}. Confidence: {confidence:.1%}')
        except Exception as e:
            messagebox.showerror('Prediction Error', str(e))

    def show_image(self, image: Image.Image):
        image = image.resize((1150, 650), Image.BILINEAR)
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.config(image=self.photo_image)

    def on_close(self):
        self.destroy()


def main():
    app = CorrosionDetectionApp()
    app.mainloop()


if __name__ == '__main__':
    main()