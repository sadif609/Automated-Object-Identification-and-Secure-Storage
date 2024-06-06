import cv2
import os
import numpy as np

class ObjectDetection:
    def __init__(self):
        self.orb = cv2.ORB_create()  # Initialize the ORB detector here
        self.training_data_dir = "training_data"
        if not os.path.exists(self.training_data_dir):
            os.makedirs(self.training_data_dir)
        self.trained_objects = self.load_trained_objects()
        self.current_object_name = None
        self.current_images = []

    def load_trained_objects(self):
        objects = {}
        for obj_name in os.listdir(self.training_data_dir):
            obj_dir = os.path.join(self.training_data_dir, obj_name)
            if os.path.isdir(obj_dir):
                images = [cv2.imread(os.path.join(obj_dir, f)) for f in os.listdir(obj_dir)]
                keypoints_and_descriptors = [self.orb.detectAndCompute(img, None) for img in images]
                objects[obj_name] = keypoints_and_descriptors
        return objects

    def detect_object(self, frame):
        # Convert the frame to grayscale for ORB detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_frame, des_frame = self.orb.detectAndCompute(gray_frame, None)

        detected_object_name = "unknown"
        max_matches = 0

        # BFMatcher with default params
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        for obj_name, kp_des_list in self.trained_objects.items():
            for kp_train, des_train in kp_des_list:
                if des_train is not None and des_frame is not None:
                    matches = bf.match(des_train, des_frame)
                    matches = sorted(matches, key=lambda x: x.distance)
                    if len(matches) > max_matches:
                        max_matches = len(matches)
                        detected_object_name = obj_name

        return detected_object_name

    def start_training(self, object_name):
        self.current_object_name = object_name
        self.current_images = []

    def capture_frame(self, frame):
        self.current_images.append(frame)
        print(f"[INFO] captured frame for {self.current_object_name}")

    def save_training_data(self):
        if self.current_object_name:
            obj_dir = os.path.join(self.training_data_dir, self.current_object_name)
            if not os.path.exists(obj_dir):
                os.makedirs(obj_dir)
            for idx, img in enumerate(self.current_images):
                img_path = os.path.join(obj_dir, f"{idx}.png")
                cv2.imwrite(img_path, img)
                print(f"[INFO] saved {img_path}")
            self.trained_objects = self.load_trained_objects()

    def list_trained_objects(self):
        return list(self.trained_objects.keys())
