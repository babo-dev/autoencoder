from models import builder
import utils
from PIL import Image
from torchvision import transforms

import streamlit as st


def load_model():
    path = "results/l040.pth"
    return utils.load_dict(path, builder.BuildAutoEncoder('simple'))


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the input size of the autoencoder
        transforms.ToTensor(),  # Convert to tensor with values in [0, 1]
    ])

    image = preprocess(image)
    image = image.unsqueeze(0)
    return image


def postprocess_image(image):
    postprocess = transforms.Compose([
        transforms.ToPILImage(),  # Convert tensor to PIL Image
        transforms.Resize((224, 224)),  # Resize back to original size
    ])

    image = image.squeeze(0)  # Remove the batch dimension
    image = postprocess(image)
    return image


def main():
    st.title("Autoencoder Image Reconstruction")

    model = load_model()

    st.write("Upload an image to see the autoencoder reconstruction.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        st.subheader("Original Image")
        st.image(original_image, caption="Uploaded Image", use_column_width=True)

        preprocessed_image = preprocess_image(original_image)

        st.write("Preprocessed Image Shape:", preprocessed_image.shape)
        print(preprocessed_image)

        if st.button("Encode and Decode"):
            reconstructed_image = model(preprocessed_image)
            reconstructed_image = postprocess_image(reconstructed_image)

            st.subheader("Reconstructed Image")
            st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)

        # Show True Image button
        if st.button("Show True Image"):
            st.subheader("True Image")
            st.image(original_image, caption="True Image", use_column_width=True)


if __name__ == "__main__":
    main()
