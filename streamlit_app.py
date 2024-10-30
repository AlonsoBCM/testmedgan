import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# Asegurarse de que Python puede encontrar la carpeta models
sys.path.append(os.path.abspath("./models"))
from models.create_model import create_model  # Importar la función para crear el modelo desde la carpeta models

# Configuración inicial de Streamlit
st.title("Transformación de Imágenes a Estilo Ultrasonido usando CycleGAN")
st.write("Sube una imagen y observa cómo CycleGAN la convierte a un estilo ultrasonido.")

# Subida de imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

# Verifica si se ha subido una imagen
if uploaded_file is not None:
    # Abre la imagen
    input_image = Image.open(uploaded_file).convert('RGB')
    st.image(input_image, caption='Imagen Original', use_column_width=True)

    # Carga del modelo
    st.write("Cargando el modelo...")

    # Configuración del modelo
    class Opt:
        def __init__(self):
            self.model = 'cycle_gan'
            self.checkpoints_dir = './modelos'  # Carpeta donde están los pesos
            self.name = 'cyclegan_ultrasonido'  # Nombre ficticio para configuración
            self.gpu_ids = [0] if torch.cuda.is_available() else []
            self.isTrain = False

    opt = Opt()
    model = create_model(opt)
    model.setup(opt)

    # Cargar los pesos del generador desde la carpeta modelos
    model.netG_A.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, 'latest_net_G_A.pth'), map_location='cpu'))

    # Transformaciones de la imagen para adaptarla a la entrada del modelo
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    input_tensor = transform(input_image).unsqueeze(0)  # Añadir batch dimension
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    # Generación de la imagen de salida
    st.write("Generando imagen en estilo ultrasonido...")
    with torch.no_grad():
        fake_image = model.netG_A(input_tensor)  # Gener

