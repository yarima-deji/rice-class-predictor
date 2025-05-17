import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image

# Load the TFLite model
tflite_model = tf.lite.Interpreter('model_vgg16_quant.tflite')
tflite_model.allocate_tensors()
input_details = tflite_model.get_input_details()
output_details = tflite_model.get_output_details()

# Class labels
CLASS_LABELS = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']


def classify_image(image_path, model):
    """
    Returns probabilities for each class, shape=(n_classes,)
    """
    img = load_img(image_path, target_size=(175, 175))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    img_array -= [0.485, 0.456, 0.406]
    img_array /= [0.229, 0.224, 0.225]

    model.set_tensor(input_details[0]['index'], img_array)
    model.invoke()
    output_data = model.get_tensor(output_details[0]['index'])[0]
    probabilities = tf.nn.softmax(output_data).numpy()
    return probabilities


def home():
    st.title("Welcome to Rice Class Predictor!")
    st.write("Welcome to Rice Class Predictor, an interactive website that leverages machine learning to classify rice grain images.")
    st.write("Upload an image and click 'Classify' to see the top predictions.")

    # Initialize history in session_state
    if 'history' not in st.session_state:
        st.session_state.history = []

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Classify"):
            with st.spinner('Classifying...'):
                probs = classify_image(uploaded_file, tflite_model)

            top_idxs = np.argsort(probs)[-3:][::-1]
            top_scores = [probs[i] for i in top_idxs]

            # Confidence threshold warning
            if top_scores[0] < 0.5:
                st.warning(f"⚠️ Low confidence: top prediction is {top_scores[0]*100:.2f}%. Results may be unreliable.")
            else:
                st.success(f"High confidence: top prediction is {top_scores[0]*100:.2f}%.")

            st.subheader("Top 3 Predictions:")
            for idx in top_idxs:
                st.write(f"{CLASS_LABELS[idx]}: {probs[idx]*100:.2f}%")

            df_probs = pd.DataFrame({
                'Class': [CLASS_LABELS[i] for i in top_idxs],
                'Probability': top_scores
            }).set_index('Class')
            st.bar_chart(df_probs)

            # Add to history
            st.session_state.history.append({
                'image': image.copy(),  # store PIL image copy
                'predictions': [(CLASS_LABELS[i], top_scores[j]) for j, i in enumerate(top_idxs)]
            })

    # Display history
    if st.session_state.history:
        st.markdown("---")
        st.subheader("History of Predictions")
        for entry in st.session_state.history:
            cols = st.columns([1, 3])
            cols[0].image(entry['image'], width=100)
            text = "\n".join([f"{label}: {score*100:.2f}%" for label, score in entry['predictions']])
            cols[1].write(text)


def about():
    st.title("About Rice Class Predictor")
    st.write("Rice Class Predictor is a state-of-the-art website that offers a reliable and accurate way to identify various types of rice grains. This website employs advanced machine learning algorithms to classify your uploaded images of rice grains, allowing you to learn more about the different varieties of rice.")
    st.write("This machine learning model has been trained on a vast dataset of high-quality rice images from all corners of the world. This ensures that the model can effectively classify rice grains of different shapes, sizes, and colors accurately.")
    st.write("So, whether you are a rice enthusiast, a food lover, or just curious about the world's different rice types, Rice Class Predictor is here to help you identify and learn more about them.")
    st.write("Final Year Project by Eaman.")


def main():
    st.set_page_config(page_title="Rice Class Classification", page_icon="🍚", layout="wide")
    st.title("Rice Class Predictor")
    st.markdown("**Project by Eaman**")

    st.sidebar.title("Navigation")
    pages = {"Home": home, "About": about}
    selection = st.sidebar.radio("Go to", list(pages.keys()))

    pages[selection]()

if __name__ == '__main__':
    main()
