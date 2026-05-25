import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import cv2

# Load the trained model
model = tf.keras.models.load_model('cnn_model.h5')

# Define the canvas size and brush width
CANVAS_SIZE = 280
BRUSH_WIDTH = 15

# Create a function to preprocess the image
def preprocess_image(image):
    image = image.resize((28, 28))
    image = image.convert('L')
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

# Function to create a blank canvas
def create_canvas():
    canvas = Image.new('RGB', (CANVAS_SIZE, CANVAS_SIZE), 'white')
    draw = ImageDraw.Draw(canvas)
    return canvas, draw

# Main Streamlit application
def main():
    st.title("SketchXAI: Sketch Recognition")
    
    st.sidebar.title("Options")
    clear_button = st.sidebar.button("Clear")
    
    # Create a canvas
    if 'canvas' not in st.session_state:
        st.session_state.canvas, st.session_state.draw = create_canvas()
    
    canvas = st.session_state.canvas
    draw = st.session_state.draw
    
    # Display the canvas
    st.image(canvas, caption="Draw on the canvas", use_column_width=True)
    
    # Draw on the canvas
    if st.sidebar.checkbox("Draw"):
        st.write("Draw using your mouse")
        # Get the coordinates of the mouse click
        x, y = st.slider("x"), st.slider("y")
        if st.sidebar.button("Draw"):
            draw.rectangle([x, y, x+BRUSH_WIDTH, y+BRUSH_WIDTH], fill="black")
            st.image(canvas, caption="Draw on the canvas", use_column_width=True)
    
    # Predict the drawing
    if st.sidebar.button("Predict"):
        processed_image = preprocess_image(canvas)
        prediction = model.predict(processed_image)
        predicted_class = np.argmax(prediction)
        st.write(f"Predicted Class: {predicted_class}")
    
    # Clear the canvas
    if clear_button:
        st.session_state.canvas, st.session_state.draw = create_canvas()
        st.image(st.session_state.canvas, caption="Draw on the canvas", use_column_width=True)

if __name__ == "__main__":
    main()
