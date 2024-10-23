import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO

# Clase para el modelo de CycleGAN
class CycleGANModel:
    def _init_(self, model_data):
        # Cargar el modelo desde los datos proporcionados
        try:
            state_dict = torch.load(model_data, map_location=torch.device('cpu'))

            # Crear una instancia del modelo base
            self.model = self.build_model()

            # Cargar los pesos en el modelo
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            st.error(f"Error al cargar los pesos del modelo: {e}")
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

# Configuración de la página de Streamlit
st.title("Transformación de Imágenes con CycleGAN")

# Cargar el modelo
model_file = st.file_uploader("Sube tu modelo de CycleGAN (.pth)", type=["pth"])

# Cargar la imagen
uploaded_image = st.file_uploader("Sube una imagen (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])

if model_file and uploaded_image:
    # Mostrar la imagen original y sus dimensiones
    image = Image.open(uploaded_image)
    st.image(image, caption=f"Imagen original - Dimensiones: {image.size}", use_column_width=True)

    # Redimensionar la imagen a 256x256
    resized_image = image.resize((256, 256))
    st.image(resized_image, caption=f"Imagen redimensionada a 256x256 - Dimensiones: {resized_image.size}", use_column_width=True)

    # Instanciar el modelo de CycleGAN
    model = CycleGANModel(model_file)

    # Transformar la imagen con el modelo cargado
    transformed_image = model.transform(resized_image)

    if transformed_image:
        st.image(transformed_image, caption="Imagen transformada por CycleGAN", use_column_width=True)

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
elif model_file is None:
    st.info("Por favor, sube un modelo de CycleGAN en formato .pth")
elif uploaded_image is None:
    st.info("Por favor, sube una imagen en formato .jpg, .jpeg o .png")
