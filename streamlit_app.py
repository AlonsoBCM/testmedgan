import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import requests
from io import BytesIO

# Clase para el modelo de CycleGAN
class CycleGANModel:
    def _init_(self, model_url):
        # Descargar el modelo desde la URL proporcionada
        response = requests.get(model_url, stream=True)

        if response.status_code == 200:
            model_data = BytesIO(response.content)

            # Cargar solo los pesos del modelo para evitar problemas de registro de clases
            try:
                state_dict = torch.load(model_data, map_location=torch.device('cpu'), weights_only=True)

                # Crear una instancia del modelo base
                self.model = self.build_model()

                # Cargar los pesos en el modelo
                self.model.load_state_dict(state_dict)
                self.model.eval()
            except Exception as e:
                st.error(f"Error al cargar los pesos del modelo: {e}")
                self.model = None
        else:
            st.error(f"Error al descargar el modelo: {response.status_code}")
            self.model = None

    def build_model(self):
        """
        Define la arquitectura base del modelo aquí.
        Debes adaptar esta parte según la estructura de tu modelo de CycleGAN.
        """
        from torch import nn

        class SimpleCycleGAN(nn.Module):
            def _init_(self):
                super(SimpleCycleGAN, self)._init_()
                # Define aquí la arquitectura de tu modelo. 
                # Esto es solo un placeholder; ajusta la arquitectura según tu modelo entrenado.
                self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)

            def forward(self, x):
                return self.conv(x)

        return SimpleCycleGAN()

    def transform(self, image):
        if self.model is None:
            return None

        # Transformar la imagen de PIL a tensor
        transform_to_tensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image_tensor = transform_to_tensor(image).unsqueeze(0)

        # Aplicar la transformación usando el modelo de CycleGAN
        with torch.no_grad():
            transformed_tensor = self.model(image_tensor)[0]

        # Convertir el tensor transformado de nuevo a imagen PIL
        transform_to_image = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2)),
            transforms.ToPILImage()
        ])
        transformed_image = transform_to_image(transformed_tensor)

        return transformed_image

# URL del modelo de CycleGAN desde GitHub (raw)
model_url = "https://raw.githubusercontent.com/AlonsoBCM/testmedgan/main/modelos/modelo_entrenado.pth"

# Intentar cargar el modelo de CycleGAN
st.title("Transformación de imágenes con CycleGAN")

try:
    model = CycleGANModel(model_url)

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

        if transformed_image:
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
        else:
            st.error("No se pudo transformar la imagen debido a un problema con el modelo.")
except Exception as e:
    st.error(f"Error inesperado: {e}")
