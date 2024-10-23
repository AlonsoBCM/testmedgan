import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests
from io import BytesIO

# Definir el modelo de CycleGAN
class CycleGANModel:
    def _init_(self, model_url):
        # Descargar el modelo desde GitHub
        response = requests.get(model_url)
        response.raise_for_status()  # Verifica si la descarga fue exitosa
        model_data = BytesIO(response.content)
        self.model = torch.load(model_data, map_location=torch.device('cpu'))
        self.model.eval()

    def transform(self, image):
        # Transformar la imagen de PIL a tensor
        transform_to_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform_to_tensor(image).unsqueeze(0)

        # Transformar la imagen usando el modelo de CycleGAN
        with torch.no_grad():
            transformed_tensor = self.model(image_tensor)[0]

        # Convertir el tensor transformado de nuevo a imagen PIL
        transform_to_image = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
            transforms.ToPILImage()
        ])
        transformed_image = transform_to_image(transformed_tensor)

        return transformed_image

# URL del modelo de CycleGAN
model_url = "https://github.com/AlonsoBCM/testmedgan/raw/main/modelos/modelo_entrenado.pth"

try:
    # Cargar el modelo de CycleGAN
    model = CycleGANModel(model_url)

    # Configuración de la página de Streamlit
    st.title("Transformación de imágenes con CycleGAN")

    # Cargar la imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Mostrar la imagen original y sus dimensiones
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Imagen original - Dimensiones: {image.size}", use_column_width=True)

        # Redimensionar la imagen a 256x256
        resized_image = image.resize((256, 256))
        st.image(resized_image, caption=f"Imagen redimensionada a 256x256 - Dimensiones: {resized_image.size}", use_column_width=True)

        # Transformar la imagen con CycleGAN
        transformed_image = model.transform(resized_image)
        st.image(transformed_image, caption="Imagen transformada por CycleGAN (A to B)", use_column_width=True)

        # Botón para descargar la imagen transformada
        buf = BytesIO()
        transformed_image.save(buf, format='PNG')
        byte_im = buf.getvalue()

        st.download_button(
            label="Descargar imagen transformada",
            data=byte_im,
            file_name="imagen_transformada.png",
            mime="image/png"
        )
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
