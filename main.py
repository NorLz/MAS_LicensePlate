import time
import streamlit as st
from utils.image import extract_from_img
import cv2
from PIL import Image
from utils.image import crop_license_plate, preprocess_image
from utils.ocr import ocr
from utils.model import model
import tempfile

import time
import streamlit as st
from utils.image import extract_from_img, crop_license_plate, preprocess_image
import cv2
from PIL import Image
from utils.ocr import ocr
from utils.model import model
import numpy as np
def main():
    st.set_page_config(
        page_title="License Plate Project",
        page_icon=":sunglasses:",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    st.title("License Plate Recognition and Text Extraction")
    tab1, tab2 = st.tabs(["Upload Image", "Open your Camera"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image")
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded image")
            
            # Display progress bar
            progress_bar = st.progress(0)
            st.text("Processing the image, please wait...")

            # Simulate progress while processing
            for percent_complete in range(0, 100, 10):
                time.sleep(0.1)  # Simulating processing delay
                progress_bar.progress(percent_complete)
            
            # Process the image
            result = extract_from_img(uploaded_file)

            # Complete the progress bar
            progress_bar.progress(100)
            
            # Display the result
            st.write(result)
            st.snow()

    with tab2:
        st.write("Camera")
        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])  # Placeholder for displaying frames
        
        # Load model instance
        model_instance = model()
        ocr_instance = ocr()

        # Open the camera feed if the checkbox is checked
        if run:
            cap = cv2.VideoCapture(0)  # Open default camera
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera.")
                    break

                # Convert to RGB for Streamlit display
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform prediction on the frame
                temp_file_path = "temp_frame.jpg"
                cv2.imwrite(temp_file_path, frame)
                predictions = model_instance.predict(temp_file_path, confidence=40, overlap=30).json()

                if 'predictions' in predictions and len(predictions['predictions']) > 0:
                    for plate in predictions['predictions']:
                        # Get bounding box coordinates
                        x_center, y_center = plate['x'], plate['y']
                        width, height = plate['width'], plate['height']
                        x_min = int(x_center - width / 2)
                        y_min = int(y_center - height / 2)
                        x_max = int(x_center + width / 2)
                        y_max = int(y_center + height / 2)

                        # Draw bounding box
                        cv2.rectangle(rgb_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Crop and preprocess the license plate area
                        cropped_frame = frame[y_min:y_max, x_min:x_max]
                        if cropped_frame.size != 0:
                            pil_cropped = Image.fromarray(cropped_frame)
                            processed_image = preprocess_image(pil_cropped)
                            
                            # Perform OCR
                            text_result = ocr_instance.ocr(processed_image, det=False, cls=False)
                            extracted_text = text_result[0][0][0] if text_result else "No Text Detected"
                            
                            # Display extracted text above the bounding box
                            cv2.putText(rgb_frame, extracted_text, (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Update the frame in the Streamlit app
                FRAME_WINDOW.image(rgb_frame)

            cap.release()
        else:
            st.write("Click the checkbox to start the camera.")

    
if __name__ == "__main__":
    main()