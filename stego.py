import os
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Step 1: Feature Extraction using LSB
def extract_lsb_features(image_path):
    """
    Extract LSB features from a resized image for classification.
    """
    image = Image.open(image_path).resize((128, 128)).convert('L')
    pixels = np.array(image)
    lsb = pixels & 1  # Extract the least significant bit
    return lsb.flatten()

# Step 2: Decode Hidden Message
def decode_hidden_message(image_path):
    """
    Decode the hidden message from an image by extracting LSB bits.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    pixels = np.array(image).flatten()
    lsb_bits = [str(pixel & 1) for pixel in pixels]

    # Group bits into bytes (8 bits each)
    binary_message = ''.join(lsb_bits)
    bytes_list = [binary_message[i:i+8] for i in range(0, len(binary_message), 8)]

    # Convert bytes to characters until a null terminator is found
    message = ""
    for byte in bytes_list:
        char = chr(int(byte, 2))
        if char == '\x00':  # Null terminator
            break
        message += char

    return message

# Step 3: Load and Balance Dataset
def load_and_balance_dataset(cover_folder, stego_folder):
    """
    Load cover and stego images, extract LSB features, and balance the dataset.
    """
    features, labels = [], []
    for folder, label in [(cover_folder, 0), (stego_folder, 1)]:
        for filename in os.listdir(folder):
            if filename.endswith('.png'):
                image_path = os.path.join(folder, filename)
                features.append(extract_lsb_features(image_path))
                labels.append(label)

    # Balance dataset using RandomUnderSampler
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(np.array(features), np.array(labels))

# Step 4: Train Model
def train_model(cover_folder, stego_folder):
    """
    Train a Random Forest model to detect steganographic images.
    """
    # Load and balance the dataset
    features, labels = load_and_balance_dataset(cover_folder, stego_folder)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features)

    # Train Random Forest model
    classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    classifier.fit(X_train, labels)

    # Save model and scaler
    joblib.dump(classifier, "steganography_detection_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    return classifier, scaler

# Step 5: Streamlit Interface
def main():
    st.title("Steganography Detection and Message Decoding")

    # Input for folder paths to train the model
    cover_folder = st.text_input("Path to Cover Folder", "path/to/cover/folder")
    stego_folder = st.text_input("Path to Stego Folder", "path/to/stego/folder")
    
    if st.button("Train Model"):
        classifier, scaler = train_model(cover_folder, stego_folder)
        st.success("Model trained and saved!")

    # Upload image for prediction
    uploaded_image = st.file_uploader("Upload an image for prediction", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        # Display the image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Load the trained model and scaler
        if os.path.exists("steganography_detection_model.pkl") and os.path.exists("scaler.pkl"):
            classifier = joblib.load("steganography_detection_model.pkl")
            scaler = joblib.load("scaler.pkl")

            # Feature extraction and prediction
            features = extract_lsb_features(uploaded_image).reshape(1, -1)
            features = scaler.transform(features)
            prediction = classifier.predict(features)[0]

            if prediction == 1:
                st.subheader("The image is steganographic.")
                hidden_message = decode_hidden_message(uploaded_image)
            else:
                st.subheader("The image is not steganographic.")
        else:
            st.error("Model is not trained yet. Please train the model first.")

if __name__ == "__main__":
    main()
