import streamlit as st
from PIL import Image
import io

# from torchvision import transforms  # Comentado ya que es parte de CycleGAN

# Placeholder para el modelo de CycleGAN
# class CycleGANModel:
#     def _init_(self, model_path):
#         self.model = torch.load(model_path, map_location=torch.device('cpu'))
#         self.model.eval()

#     def transform(self, image):
#         with torch.no_grad():
#             transformed_image = self.model(image)
#         return transformed_image

# Función para redimensionar la imagen
def resize_image(image, size=(256, 256)):
    return image.resize(size)

# # Función para convertir la imagen a tensor
# def image_to_tensor(image):
#     transform = transforms.ToTensor()
#     return transform(image).unsqueeze(0)

# # Función para convertir el tensor a imagen
# def tensor_to_image(tensor):
#     transform = transforms.ToPILImage()
#     image = transform(tensor.squeeze(0))
#     return image

# Configuración de la página de Streamlit
st.title("Transformación de imágenes con CycleGAN")

# Carga de la imagen
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Abrir y redimensionar la imagen
    image = Image.open(uploaded_file)
    resized_image = resize_image(image)

    # Mostrar la imagen redimensionada
    st.image(resized_image, caption='Imagen redimensionada a 256x256', use_column_width=True)

    # Convertir la imagen redimensionada a tensor (comentado para CycleGAN)
    # image_tensor = image_to_tensor(resized_image)

    # Placeholder para cargar el modelo de CycleGAN (comentado)
    # model_path = 'path_a_tu_modelo/cyclegan_modelo.pt'  # Cambia este path a la ubicación de tu modelo
    # model = CycleGANModel(model_path)

    # Transformar la imagen con el modelo de CycleGAN (comentado)
    # transformed_tensor = model.transform(image_tensor)

    # Convertir el tensor transformado a imagen (comentado)
    # transformed_image = tensor_to_image(transformed_tensor)

    # Mostrar la imagen transformada (comentado)
    # st.image(transformed_image, caption='Imagen transformada por CycleGAN', use_column_width=True)

    # Botón para descargar la imagen transformada (comentado)
    # buf = io.BytesIO()
    # transformed_image.save(buf, format='PNG')
    # byte_im = buf.getvalue()

    # st.download_button(
    #     label="Descargar imagen transformada",
    #     data=byte_im,
    #     file_name="imagen_transformada.png",
    #     mime="image/png"
    # )
