import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10  # Example dataset
import cv2

# Step 2: Define a function to train the model
def train_model():
    # Load example data (replace this with your dataset and preprocessing code)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    
    # Define a simple model (adjust architecture as needed)
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    return model

# Step 3: Define a prediction function
def predict_action(model, file):
    # Read file as an image
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Preprocess the image (adjust size as needed for your model)
    frame = cv2.resize(frame, (32, 32))  # Example size
    frame = frame / 255.0
    frame = frame.reshape((1, 32, 32, 3))  # Adjust dimensions as per model
    
    # Make prediction
    prediction = model.predict(frame)
    action = np.argmax(prediction)
    return action

# Streamlit App
st.title("Human Action Recognition")

# Step 4: Button to start training
if st.button("Train Model"):
    with st.spinner("Training the model, please wait..."):
        model = train_model()
        st.success("Model trained successfully!")

# Step 5: File uploader for testing the model
uploaded_file = st.file_uploader("Upload an image or video for action prediction", type=["jpg", "jpeg", "png", "mp4"])

# Step 6: Predict action if model is trained and file is uploaded
if uploaded_file is not None:
    if 'model' not in locals():
        st.warning("Please train the model first!")
    else:
        action = predict_action(model, uploaded_file)
        st.write(f"Predicted Action: {action}")
