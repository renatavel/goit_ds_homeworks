import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16
from PIL import Image

(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
X_train_exp = np.expand_dims(X_train, -1)
X_test_exp = np.expand_dims(X_test, -1)

class_names = ["T-shirt", "Trousers", "Pullover", "Dress", "Coat", "Sandals", "Shirt", "Sneaker", "Bag", "Ankle boots"]

def cnn_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def vgg16_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-4:]:
        layer.trainable = True
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

st.title("Classification of the images from Fashion MNIST dataset")

st.markdown("### Choose the model for classification:")
model_choice = st.selectbox("", ("Custom CNN", "VGG16"))

st.markdown('### Upload the image in either PNG format, or JPG:')
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    image_resized = image.resize((28, 28))
    input_arr = np.array(image_resized) / 255.0
    st.image(image, caption="Uploaded image", use_container_width=True)

    if model_choice == "Custom CNN":
        model = cnn_model()
        model.load_weights("streamlit_files/cnn_weights.h5")
        input_arr = np.expand_dims(input_arr, axis=(0, -1))
    else:
        model = vgg16_model()
        model.load_weights("streamlit_files/vgg16_weights.h5")
        input_arr = np.expand_dims(input_arr, axis=-1)
        input_arr = tf.convert_to_tensor(input_arr)
        input_arr = tf.image.grayscale_to_rgb(input_arr)
        input_arr = tf.image.resize(input_arr, (32, 32))
        input_arr = tf.expand_dims(input_arr, axis=0)

    prediction = model.predict(input_arr)[0]
    predicted_class = np.argmax(prediction)

    st.subheader("Results of classification:")
    results = {
        "**Class Names**": class_names,
        "**Probability**": [f"{p:.2f}" for p in prediction]
    }
    st.table(results)
    st.markdown(f"### Predicted class: <span style='font-weight: normal'>{class_names[predicted_class]}</span>", unsafe_allow_html=True)

if st.checkbox("Show visualization of model's loss and accuracy"):
    history = np.load("streamlit_files/history_cnn.npy", allow_pickle=True).item() if model_choice == "Custom CNN" else np.load("streamlit_files/history_vgg.npy", allow_pickle=True).item()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['loss'], label='train loss')
    ax1.plot(history['val_loss'], label='val loss')
    ax1.set_title("Loss function")
    ax1.legend()
    ax2.plot(history['accuracy'], label='train acc')
    ax2.plot(history['val_accuracy'], label='val acc')
    ax2.set_title("Accuracy")
    ax2.legend()
    st.pyplot(fig)
