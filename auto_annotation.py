import os
import shutil
import supervision as sv
import cv2
import numpy as np
import matplotlib.pyplot as plt
from autodistill.detection import CaptionOntology
from autodistill_grounded_sam import GroundedSAM

# Update the paths to point to the local directories
IMAGE_DIR_PATH = 'C:/Users/EdutopiaLabs/Desktop/GitHub-Deepali/SAMv2-YOLO11/satellite_images'  # Local path for satellite images
DATASET_DIR_PATH = 'C:/Users/EdutopiaLabs/Desktop/GitHub-Deepali/SAMv2-YOLO11/annoted_dataset'  # Local path for annotated dataset
VALID_IMAGES_PATH = 'C:/Users/EdutopiaLabs/Desktop/GitHub-Deepali/SAMv2-YOLO11/annoted_dataset/valid/images'
VALID_LABELS_PATH = 'C:/Users/EdutopiaLabs/Desktop/GitHub-Deepali/SAMv2-YOLO11/annoted_dataset/valid/labels'
TEST_IMAGES_PATH = 'C:/Users/EdutopiaLabs/Desktop/GitHub-Deepali/SAMv2-YOLO11/annoted_dataset/test/images'
TEST_LABELS_PATH = 'C:/Users/EdutopiaLabs/Desktop/GitHub-Deepali/SAMv2-YOLO11/annoted_dataset/test/labels'


# Set up ontology and labeling model
ontology = CaptionOntology({
    "Flat Roof Tops": "rooftops",
})

base_model = GroundedSAM(ontology=ontology)
dataset = base_model.label(input_folder=IMAGE_DIR_PATH, extension=".jpg", output_folder=DATASET_DIR_PATH)

# Move files from valid to test
os.makedirs(TEST_IMAGES_PATH, exist_ok=True)
os.makedirs(TEST_LABELS_PATH, exist_ok=True)

image_files = os.listdir(VALID_IMAGES_PATH)
image_files = [f for f in image_files if f.endswith('.jpg')]

num_files_to_move = 6
files_to_move = image_files[:num_files_to_move]

for image_file in files_to_move:
    source_image_path = os.path.join(VALID_IMAGES_PATH, image_file)
    dest_image_path = os.path.join(TEST_IMAGES_PATH, image_file)
    shutil.move(source_image_path, dest_image_path)

    label_file = image_file.replace('.jpg', '.txt')
    source_label_path = os.path.join(VALID_LABELS_PATH, label_file)
    dest_label_path = os.path.join(TEST_LABELS_PATH, label_file)

    if os.path.exists(source_label_path):
        shutil.move(source_label_path, dest_label_path)
    else:
        print(f"Warning: Label file {label_file} not found for image {image_file}")

print(f"Moved {num_files_to_move} images and their corresponding labels from valid to test folder.")