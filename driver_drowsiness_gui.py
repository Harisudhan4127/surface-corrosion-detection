import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk


class DrowsinessDetector:
    def __init__(self, closed_frame_threshold: int = 20):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml'
        )
        self.closed_frame_threshold = closed_frame_threshold
        self.closed_frames = 0

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError('Unable to load OpenCV Haar cascades. Make sure OpenCV is installed correctly.')

    def analyze_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str, bool]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120),
        )

        status = 'No face detected'
        drowsy = False

        if len(faces) == 0:
            self.closed_frames = max(0, self.closed_frames - 1)
        else:
            for (x, y, w, h) in faces:
                face_gray = gray[y:y + h, x:x + w]
                face_color = frame[y:y + h, x:x + w]

                eyes = self.eye_cascade.detectMultiScale(
                    face_gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(20, 10),
                )

                eye_count = len(eyes)
                if eye_count < 2:
                    self.closed_frames += 1
                else:
                    self.closed_frames = max(0, self.closed_frames - 2)

                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        face_color,
                        (ex, ey),
                        (ex + ew, ey + eh),
                        (255, 220, 0),
                        2,
                    )

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 180, 255), 2)
                cv2.putText(
                    frame,
                    f'Eyes: {eye_count}',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

        if self.closed_frames >= self.closed_frame_threshold:
            drowsy = True
            status = 'DROWSY ALERT'
        elif len(faces) > 0:
            status = 'AWAKE'
        else:
            status = 'NO FACE'

        bg_color = (200, 50, 50) if drowsy else (40, 160, 40)
        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            bg_color,
            3,
        )

        return frame, status, drowsy

    def analyze_image(self, frame: np.ndarray) -> tuple[np.ndarray, str, bool]:
        current_closed = self.closed_frames
        annotated, status, drowsy = self.analyze_frame(frame)
        self.closed_frames = current_closed
        return annotated, status, drowsy


class DriverDrowsinessApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Driver Drowsiness Detection')
        self.protocol('WM_DELETE_WINDOW', self.on_close)
        self.geometry('1180x780')
        self.resizable(False, False)

        self.detector = DrowsinessDetector()
        self.capture = None
        self.video_source = None
        self.image_path = None
        self.is_streaming = False
        self.photo_image = None

        self.create_widgets()

    def create_widgets(self):
        control_frame = tk.Frame(self, padx=12, pady=8)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Button(control_frame, text='Open Image', command=self.open_image, width=15).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text='Predict Image', command=self.predict_image, width=15).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text='Start Webcam', command=self.start_webcam, width=15).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text='Start Stream', command=self.start_stream, width=15).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text='Stop', command=self.stop_stream, width=15).pack(side=tk.LEFT, padx=4)

        tk.Label(control_frame, text='Stream URL / Camera ID:', padx=8).pack(side=tk.LEFT)
        self.url_entry = tk.Entry(control_frame, width=28)
        self.url_entry.insert(0, '0')
        self.url_entry.pack(side=tk.LEFT, padx=4)

        self.status_label = tk.Label(self, text='Ready to predict', font=('Arial', 16), bg='#202020', fg='white', padx=12, pady=8)
        self.status_label.pack(side=tk.TOP, fill=tk.X)

        self.canvas = tk.Label(self, bg='black')
        self.canvas.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.info_label = tk.Label(self, text='Tips: Load an image, predict, or start a webcam/stream session.', font=('Arial', 12), fg='#f0f0f0', bg='#101010', anchor='w')
        self.info_label.pack(side=tk.BOTTOM, fill=tk.X)

    def open_image(self):
        path = filedialog.askopenfilename(
            title='Select image',
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp')],
        )
        if not path:
            return

        self.image_path = path
        self.stop_stream()
        image = cv2.imread(path)
        if image is None:
            messagebox.showerror('Error', 'Unable to load image file.')
            return

        self.show_frame(image)
        self.status_label.config(text='Image loaded: ' + os.path.basename(path), bg='#1f3b6f')

    def predict_image(self):
        if not self.image_path:
            messagebox.showwarning('Predict image', 'Please load an image first.')
            return

        image = cv2.imread(self.image_path)
        annotated, status, drowsy = self.detector.analyze_image(image)
        self.show_frame(annotated)
        self.status_label.config(
            text=f'Image prediction: {status}',
            bg='#8b1e3f' if drowsy else '#1f6927',
        )
        self.info_label.config(text='Prediction complete. Drowsy state detected.' if drowsy else 'Prediction complete. Driver appears awake.')

    def start_webcam(self):
        self.url_entry.delete(0, tk.END)
        self.url_entry.insert(0, '0')
        self.start_stream()

    def start_stream(self):
        if self.is_streaming:
            return

        source = self.url_entry.get().strip()
        if not source:
            messagebox.showwarning('Stream source', 'Enter a camera ID or stream URL.')
            return

        try:
            source_value = int(source)
        except ValueError:
            source_value = source

        self.video_source = source_value
        self.capture = cv2.VideoCapture(self.video_source, cv2.CAP_ANY)
        if not self.capture.isOpened():
            messagebox.showerror('Stream error', f'Unable to open camera or URL: {source}')
            self.capture.release()
            self.capture = None
            return

        self.is_streaming = True
        self.status_label.config(text='Streaming from source: ' + str(source), bg='#264653')
        self.update_stream()

    def update_stream(self):
        if not self.is_streaming or self.capture is None:
            return

        success, frame = self.capture.read()
        if not success:
            self.status_label.config(text='Stream disconnected or invalid frame.', bg='#8b0000')
            self.stop_stream()
            return

        annotated, status, drowsy = self.detector.analyze_frame(frame)
        self.show_frame(annotated)
        self.status_label.config(
            text=f'Stream status: {status}',
            bg='#8b1e3f' if drowsy else '#116642',
        )
        self.after(30, self.update_stream)

    def stop_stream(self):
        if self.is_streaming:
            self.is_streaming = False
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.status_label.config(text='Stream stopped', bg='#403d3d')
            self.info_label.config(text='Ready for a new prediction.')

    def show_frame(self, frame: np.ndarray):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((1150, 650), Image.BILINEAR)
        self.photo_image = ImageTk.PhotoImage(image)
        self.canvas.config(image=self.photo_image)

    def on_close(self):
        self.stop_stream()
        self.destroy()


def main():
    app = DriverDrowsinessApp()
    app.mainloop()


if __name__ == '__main__':
    main()
