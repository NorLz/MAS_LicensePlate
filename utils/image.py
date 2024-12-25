from PIL import Image, ImageEnhance
import numpy as np
from utils.ocr import ocr
from utils.model import model
from roboflow import Roboflow
import tempfile
import os


# Function to crop and save license plate from prediction
def crop_license_plate(image_path, prediction):
    # Load the image
    image = Image.open(image_path)

    # Extract bounding box coordinates
    x_center = prediction['x']
    y_center = prediction['y']
    width = prediction['width']
    height = prediction['height']

    # Calculate the bounding box (top-left and bottom-right coordinates)
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)

    # Crop the image
    cropped_image = image.crop((x_min, y_min, x_max, y_max))

    # Return the cropped image
    return cropped_image

# Function to process an image and return the cropped license plate
def predict_and_crop_image(uploaded_file):
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file_path = temp_file.name
        uploaded_file.seek(0)  # Make sure to reset file pointer
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.read())
    
    # Instantiate model
    model_instance = model()
    
    # Get predictions from the model
    prediction = model_instance.predict(temp_file_path, confidence=40, overlap=30).json()
    
    if 'predictions' in prediction and len(prediction['predictions']) > 0:
        # Use the first prediction (or loop through all if needed)
        plate_prediction = prediction['predictions'][0]

        # Crop and return the license plate
        cropped_image = crop_license_plate(temp_file_path, plate_prediction)
        return cropped_image
    else:
        print(f"No predictions for {temp_file_path}")
        return None
    

def preprocess_image(image):
    # Ensure the image is a PIL Image
    if isinstance(image, Image.Image):
        # Convert to numpy array
        image_array = np.array(image)
    else:
        # If it's already a numpy array, use it directly
        image_array = image

    # Convert to grayscale (mode 'L')
    gray_image = Image.fromarray(image_array).convert("L")

    # Enhance contrast to improve visibility
    enhancer = ImageEnhance.Contrast(gray_image)
    enhanced_image = enhancer.enhance(2.0)

    # Apply binarization (thresholding): pixels below 128 set to 0 (black), above 128 set to 255 (white)
    binarized_image = enhanced_image.point(lambda x: 0 if x < 128 else 255, 'L')

    # Return the binarized image as a numpy array
    return np.array(binarized_image)

def extract_from_img(image):
    ocr_instance = ocr()
    cropped_img = predict_and_crop_image(image)
    if cropped_img:
        # Preprocess the image
        preprocess_img = preprocess_image(cropped_img)
        
        # Perform OCR
        text_result = ocr_instance.ocr(preprocess_img, det=False, cls=False)
        print("Extracted Text:", text_result)
        return text_result
    else:
        print("No license plate detected.")