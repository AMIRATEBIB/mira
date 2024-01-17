import streamlit as st
import cv2
import numpy as np

def detect_faces(image, min_neighbors, scale_factor, rectangle_color):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if image is None:
        st.error("Error: Image not loaded properly.")
        return None
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)

    try:
        # Convert Streamlit color picker format to OpenCV format
        rectangle_color = tuple(int(i * 255) for i in rectangle_color[:3])
    except Exception as e:
        print("Error parsing color:", e)
        # Default color (blue) in case of an error with color picker
        rectangle_color = (255, 0, 0)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), rectangle_color, 2)

    return image



def main():
    st.title("Face Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image using OpenCV
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # User instructions
        st.markdown("### Instructions:")
        st.write("1. Upload an image.")
        st.write("2. Adjust parameters using the sliders.")
        st.write("3. Click the 'Detect Faces' button.")
        st.write("4. Save the result or modify parameters as needed.")

        min_neighbors = st.slider("Adjust minNeighbors:", 1, 10, 5)
        scale_factor = st.slider("Adjust scaleFactor:", 1.01, 2.0, 1.2)
        rectangle_color = st.color_picker("Choose rectangle color", "#FF5733")

        if st.button("Detect Faces"):
            result_image = detect_faces(image.copy(), min_neighbors, scale_factor, rectangle_color)
            if result_image is not None:
                st.image(result_image, channels="BGR")

                if st.button("Save Image"):
                    cv2.imwrite("result_with_faces.jpg", result_image)
                    st.success("Image saved successfully.")

if __name__ == "__main__":
    main()
