import streamlit as st
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import os
from collections import OrderedDict

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
        from your_model_architecture import YourModel  # Reemplaza con tu propia arquitectura
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

    # Convertir la imagen a blanco y negro
    image = image.convert('L')
    st.image(image, caption="Imagen en blanco y negro", use_column_width=True)

    # Crear una máscara en forma de trapecio
    trapezoid_mask = Image.new("L", (256, 256), 0)
    draw = ImageDraw.Draw(trapezoid_mask)
    # Definir los vértices del trapecio
    top_left = (64, 0)
    top_right = (192, 0)
    bottom_left = (0, 256)
    bottom_right = (256, 256)
    # Dibujar el trapecio en la máscara
    draw.polygon([top_left, top_right, bottom_right, bottom_left], fill=255)

    # Aplicar la máscara a la imagen
    image = Image.composite(image, Image.new("L", (256, 256), 0), trapezoid_mask)
    st.image(image, caption="Imagen con recorte trapezoidal", use_column_width=True)
else:
    st.warning("Por favor, sube una imagen (.png o .jpg)")

# Aplicar el modelo si ambos archivos están cargados
if uploaded_model is not None and uploaded_image is not None:
    # Transformación de imagen para ser compatible con PyTorch
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalización para una sola canal (blanco y negro)
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
