import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from modelos import Generator  # Ajustar para importar desde el directorio correcto

# Paso 1: Cargar el Modelo del Generador
# Ruta al archivo del generador preentrenado
model_path = "modelos/latest_net_G_A.pth"  # Asegúrate de proporcionar la ruta correcta al archivo

# Crear una instancia del generador
# Asegúrate de que el modelo Generator esté definido adecuadamente en tu código (similar al usado en el entrenamiento)
netG = Generator(input_nc=3, output_nc=3, ngf=64)  # Define los parámetros según la arquitectura del modelo que usaste para entrenar
netG.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
netG.eval()  # Cambiar a modo de evaluación

# Paso 2: Definir la Función para Transformar la Imagen
def transform_image(input_image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Cambia el tamaño si el modelo fue entrenado con un tamaño diferente
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normaliza como se hizo durante el entrenamiento
    ])
    return transform(input_image).unsqueeze(0)

# Paso 3: Interfaz en Streamlit
st.title("Aplicación para Transformar Imágenes a Ultrasonido con Generador G_A")

# Subir la imagen
uploaded_file = st.file_uploader("Elige una imagen para transformar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mostrar la imagen original
    input_image = Image.open(uploaded_file).convert("RGB")
    st.image(input_image, caption="Imagen Original", use_column_width=True)

    # Transformar la imagen y pasarla por el generador
    st.write("Transformando la imagen, por favor espera...")
    transformed_image = transform_image(input_image)
    
    with torch.no_grad():
        fake_image = netG(transformed_image)  # Pasar la imagen por el generador

    # Convertir la imagen de tensor a PIL para mostrar en Streamlit
    fake_image = fake_image.squeeze(0)  # Quitar el batch dimension
    fake_image = (fake_image * 0.5) + 0.5  # Desnormalizar
    fake_image = transforms.ToPILImage()(fake_image)

    # Mostrar la imagen transformada
    st.image(fake_image, caption="Imagen Transformada (Ultrasonido)", use_column_width=True)
