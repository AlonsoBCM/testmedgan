import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import os
from collections import OrderedDict

# Definir tu modelo aquí o importarlo correctamente
# Este es un placeholder. Reemplaza esta clase con la definición real de tu modelo.
class YourModel(torch.nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        # Aquí iría la definición del modelo
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

# Título de la aplicación
st.title("Subir Modelo y Probar Imagen con CycleGAN")

# Subir archivos
uploaded_model = st.file_uploader("Sube tu modelo .pth", type=["pth"])
uploaded_image = st.file_uploader("Sube tu imagen .png o .jpg", type=["png", "jpg"])

# Comprobar si se subieron los archivos
if uploaded_model is not None:
    st.success("Modelo .pth subido correctamente.")
    # Guardar el modelo subido en el sistema de archivos
    with open("model.pth", "wb") as f:
        f.write(uploaded_model.read())
    # Cargar el modelo con Torch
    try:
        state_dict = torch.load("model.pth", map_location=torch.device('cpu'))

        # Eliminar el prefijo 'module.' del state_dict si existe
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")  # Eliminar 'module.' del nombre de las claves
            new_state_dict[name] = v

        # Crear una instancia del modelo y cargar el estado
        model = YourModel()
        model.load_state_dict(new_state_dict)
        model.eval()

        st.success("Modelo cargado exitosamente.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
else:
    st.warning("Por favor, sube un archivo de modelo (.pth)")

if uploaded_image is not None:
    st.success("Imagen subida correctamente.")
    # Cargar la imagen con PIL
    image = Image.open(uploaded_image)
    # Convertir a PNG si la imagen es JPG
    if image.format == "JPEG":
        image = image.convert('RGB')
    # Redimensionar la imagen a 256x256
    image = image.resize((256, 256))
    st.image(image, caption="Imagen redimensionada (256x256)", use_column_width=True)
else:
    st.warning("Por favor, sube una imagen (.png o .jpg)")

# Aplicar el modelo si ambos archivos están cargados
if uploaded_model is not None and uploaded_image is not None:
    # Transformación de imagen para ser compatible con PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),  # Asegura que la imagen es un tensor antes de normalizar
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Aplicar transformación a la imagen
    input_image = transform(image).unsqueeze(0)  # Añadir dimensión de batch

    # Asegurarse que el modelo está en modo evaluación
    if 'model' in locals():
        try:
            # Imagen procesada por ambos lados (de ida y vuelta con CycleGAN)
            with torch.no_grad():
                generated_image_1 = model(input_image)[0]  # Primera transformación
                generated_image_2 = model(generated_image_1.unsqueeze(0))[0]  # De vuelta

            # Convertir a imagen para mostrarla
            generated_image_1_pil = transforms.ToPILImage()(generated_image_1.squeeze(0))
            generated_image_2_pil = transforms.ToPILImage()(generated_image_2.squeeze(0))

            # Mostrar las imágenes generadas
            st.image(generated_image_1_pil, caption="Imagen generada - Primera transformación", use_column_width=True)
            st.image(generated_image_2_pil, caption="Imagen generada - Segunda transformación", use_column_width=True)
        except Exception as e:
            st.error(f"Error al procesar la imagen con el modelo: {str(e)}")
    else:
        st.warning("El modelo no ha sido cargado correctamente.")
else:
    st.info("Sube ambos archivos para procesar la imagen.")
