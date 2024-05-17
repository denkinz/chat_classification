import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

# Загрузка обученной модели
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load('model_v5.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Предобработка изображений
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    return image.unsqueeze(0)

# Функция для отображения изображения
def display_image(image):
    image_np = image.permute(1, 2, 0).numpy()  # Преобразуем тензор в numpy массив
    image_np = (image_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])  # Denormalize
    image_np = np.clip(image_np, 0, 1)
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    ax.axis('off')
    st.pyplot(fig)

# Функция для предсказания
def predict(image, model):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()
    return prediction

# Streamlit интерфейс
st.title('Image Classification by KDS')
st.write('Upload an image to classify it.')

# Загрузка изображения
uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    model = load_model()
    prediction = predict(image, model)
    prediction=prediction*100
    st.write(f'Prediction: {prediction:.2f} %')

    # Отображение изображения с прогнозом
    display_image(transforms.ToTensor()(image))
