import tkinter as tk
from tkinter import messagebox
import cv2
import threading
import numpy as np
from PIL import Image, ImageTk
from object_detection_module import ObjectDetection

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Automated Object Identification & Secure Storage")
        self.root.configure(bg="#CEF979")  # Set background color
        
        self.faculty = "Faculty: Abida Sultana"
        self.course = "Course: CSE 316"
        self.author_info = "Author\nMd Ashiqul Islam , Id : 212002056\nMd Taiab , Id : 212002087"

        # Course label
        course_label = tk.Label(root, text=self.course, font=("Helvetica", 12), bg="#CEF979")
        course_label.pack(side=tk.TOP, anchor="nw")

        # Title label
        title_label = tk.Label(root, text="Automated Object Identification & Secure Storage", font=("Helvetica", 18, "bold"), pady=10, bg="#CEF979", fg="#5A5A5A")  # Set title color
        title_label.pack()

        # Faculty label
        faculty_label = tk.Label(root, text=self.faculty, font=("Helvetica", 12), bg="#CEF979")
        faculty_label.pack()

        self.cap = cv2.VideoCapture(0)
        
        self.od = ObjectDetection()
        
        self.panel = tk.Label(root)
        self.panel.pack(padx=10, pady=10)
        
        btn_frame = tk.Frame(root, bg="#CEF979")
        btn_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.detect_btn = tk.Button(btn_frame, text="Detect Object", command=self.detect_object, width=15, font=("Helvetica", 12))
        self.detect_btn.pack(side="left", padx=10, pady=10)

        self.train_btn = tk.Button(btn_frame, text="Object Data Train", command=self.train_object, width=15, font=("Helvetica", 12))
        self.train_btn.pack(side="left", padx=10, pady=10)

        self.list_btn = tk.Button(btn_frame, text="List Trained Objects", command=self.list_objects, width=15, font=("Helvetica", 12))
        self.list_btn.pack(side="left", padx=10, pady=10)

        self.exit_btn = tk.Button(btn_frame, text="Exit", command=self.on_close, width=15, font=("Helvetica", 12), bg="red", fg="white")
        self.exit_btn.pack(side="right", padx=10, pady=10)

        self.author_label = tk.Label(root, text=self.author_info, font=("Helvetica", 10), bg="#CEF979")
        self.author_label.pack(side=tk.BOTTOM, pady=10)

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def video_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)
                self.panel.configure(image=image)
                self.panel.image = image

    def detect_object(self):
        ret, frame = self.cap.read()
        if ret:
            object_name = self.od.detect_object(frame)
            messagebox.showinfo("Detected Object", f"Object: {object_name}")

    def train_object(self):
        train_window = tk.Toplevel(self.root)
        train_window.title("Train Object")
        train_window.configure(bg="#CEF979")

        name_label = tk.Label(train_window, text="Enter object name:", font=("Helvetica", 12), bg="#CEF979")
        name_label.pack(pady=10)

        self.name_entry = tk.Entry(train_window, font=("Helvetica", 12))
        self.name_entry.pack(pady=10)
        
        self.ready_btn = tk.Button(train_window, text="Ready", command=self.prepare_training, font=("Helvetica", 12))
        self.ready_btn.pack(pady=10)
        
        self.capture_btn = tk.Button(train_window, text="Capture", command=self.capture_photo, state=tk.DISABLED, font=("Helvetica", 12))
        self.capture_btn.pack(pady=10)
        
        self.save_btn = tk.Button(train_window, text="Save", command=self.save_training_data, state=tk.DISABLED, font=("Helvetica", 12))
        self.save_btn.pack(pady=10)

    def prepare_training(self):
        self.object_name = self.name_entry.get()
        if self.object_name:
            self.capture_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.od.start_training(self.object_name)

    def capture_photo(self):
        ret, frame = self.cap.read()
        if ret:
            self.od.capture_frame(frame)

    def save_training_data(self):
        self.od.save_training_data()
        messagebox.showinfo("Training", f"Training data saved for {self.object_name}")

    def list_objects(self):
        objects = self.od.list_trained_objects()
        messagebox.showinfo("Trained Objects", "\n".join(objects))

    def on_close(self):
        print("[INFO] closing...")
        self.stop_event.set()
        self.cap.release()
        self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
