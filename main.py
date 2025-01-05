import time
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
from utils.ocr import ocr
from utils.model import model
from utils.image import extract_from_img, preprocess_image
import numpy as np
import io
import base64

def image_to_base64(img_array):
    """Convert a numpy image array to a base64 string."""
    img = Image.fromarray(img_array)
    img = img.resize((100, 100))  # Resize to 100x100
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    return base64.b64encode(img_buffer.read()).decode()

def main():
    st.set_page_config(
        page_title="MAS License Plate",
        page_icon="img/usep-logo.png",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("MAS License Plate")
    tab1, tab2 = st.tabs(["Upload Image", "Open your Camera"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image")
        
        # Initialize the list to store results if not already
        if "uploaded_results" not in st.session_state:
            st.session_state["uploaded_results"] = []

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded image")
            
            # Display progress bar
            progress_bar = st.progress(0)
            st.text("Processing the image, please wait...")

            # Simulate progress while processing
            for percent_complete in range(0, 100, 10):
                time.sleep(0.1)  # Simulating processing delay
                progress_bar.progress(percent_complete)
            
            # Convert uploaded image to a NumPy array
            uploaded_img = np.array(Image.open(uploaded_file))
            
            # Perform prediction using the model
            model_instance = model()
            ocr_instance = ocr()
            temp_file_path = "uploaded_image.jpg"
            cv2.imwrite(temp_file_path, cv2.cvtColor(uploaded_img, cv2.COLOR_RGB2BGR))
            predictions = model_instance.predict(temp_file_path, confidence=40, overlap=30).json()

            # Complete the progress bar
            progress_bar.progress(100)

            # Extract data from the predictions
            if 'predictions' in predictions and len(predictions['predictions']) > 0:
                for plate in predictions['predictions']:
                    # Get bounding box coordinates
                    x_center, y_center = plate['x'], plate['y']
                    width, height = plate['width'], plate['height']
                    x_min = int(x_center - width / 2)
                    y_min = int(y_center - height / 2)
                    x_max = int(x_center + width / 2)
                    y_max = int(y_center + height / 2)

                    # Crop the license plate area
                    cropped_frame = uploaded_img[y_min:y_max, x_min:x_max]
                    if cropped_frame.size != 0:
                        pil_cropped = Image.fromarray(cropped_frame)
                        processed_image = preprocess_image(pil_cropped)
                        
                        # Perform OCR
                        text_result = ocr_instance.ocr(processed_image, det=False, cls=False)
                        extracted_text = text_result[0][0][0] if text_result else "No Text Detected"
                        confidence = text_result[0][0][1] if text_result else 0.0

                        # Get current timestamp
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                        # Add result to session state
                        st.session_state["uploaded_results"].append({
                            "License Plate": extracted_text,
                            "CL": confidence,
                            "Time": timestamp,
                            "Original Image": uploaded_img,
                            "Cropped Plate Image": cropped_frame
                        })

            # Display the table
            if st.session_state["uploaded_results"]:
                results_df = pd.DataFrame(st.session_state["uploaded_results"])
                results_df['#'] = range(1, len(st.session_state["uploaded_results"]) + 1)
                results_df['Original Image'] = results_df['Original Image'].apply(
                    lambda img: f'<img src="data:image/png;base64,{image_to_base64(img)}" width="100" height="100">'
                )
                results_df['Cropped Plate Image'] = results_df['Cropped Plate Image'].apply(
                    lambda img: f'<img src="data:image/png;base64,{image_to_base64(img)}" width="100" height="100">' if img is not None else "No Image"
                )

                # Render the table as HTML
                table_html = results_df.to_html(escape=False, index=False, columns=['#', 'License Plate', 'CL', 'Time', 'Original Image', 'Cropped Plate Image'])
                st.markdown(table_html, unsafe_allow_html=True)



    with tab2:
        st.write("Camera")
        run = st.checkbox("Start Camera")
        FRAME_WINDOW = st.image([])  # Placeholder for displaying frames
        
        # Load model and OCR instances
        model_instance = model()
        ocr_instance = ocr()

        # Initialize results table
        detected_plates = []

        # Open the camera feed if the checkbox is checked
        if run:
            cap = cv2.VideoCapture(0)  # Open default camera

            # Placeholder for the table
            table_placeholder = st.empty()

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
                            confidence = text_result[0][0][1] if text_result else 0.0

                            # Get current timestamp
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                            # Update detected plates list
                            found = False
                            for plate_entry in detected_plates:
                                if plate_entry['License Plate'] == extracted_text:
                                    if confidence > plate_entry['CL']:
                                        plate_entry['CL'] = confidence
                                        plate_entry['Time'] = timestamp
                                        plate_entry['original_image'] = rgb_frame
                                        plate_entry['cropped_image'] = cropped_frame
                                    found = True
                                    break
                            if not found:
                                detected_plates.append({
                                    "License Plate": extracted_text,
                                    "CL": confidence,
                                    "Time": timestamp,
                                    "original_image": rgb_frame,
                                    "cropped_image": cropped_frame
                                })

                            # Display extracted text above the bounding box
                            cv2.putText(rgb_frame, extracted_text, (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Update the frame in the Streamlit app
                FRAME_WINDOW.image(rgb_frame)

                # Update the table dynamically
                if detected_plates:
                    detected_plates_df = pd.DataFrame(detected_plates)
                    detected_plates_df['#'] = range(1, len(detected_plates) + 1)
                    detected_plates_df['Original Image'] = detected_plates_df['original_image'].apply(
                        lambda img: f'<img src="data:image/png;base64,{image_to_base64(img)}" width="100" height="100">'
                    )
                    detected_plates_df['Cropped Plate Image'] = detected_plates_df['cropped_image'].apply(
                        lambda img: f'<img src="data:image/png;base64,{image_to_base64(img)}" width="100" height="100">' if img is not None else "No Image"
                    )

                    # Render as HTML
                    table_html = detected_plates_df.to_html(escape=False, index=False, columns=['#', 'License Plate', 'CL', 'Time', 'Original Image', 'Cropped Plate Image'])
                    table_placeholder.markdown(table_html, unsafe_allow_html=True)

            cap.release()
        else:
            st.write("Click the checkbox to start the camera.")

    
if __name__ == "__main__":
    main()
