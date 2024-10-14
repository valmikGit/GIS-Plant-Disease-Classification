from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.models import load_model
from django.http import JsonResponse
from PIL import Image
import numpy as np
import os
import uuid
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import json
import base64
import cv2
from django.http import HttpRequest, HttpResponse

# Load the model (this can be done once when the server starts)
ml_model = load_model(
    r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\GIS-Plant-Disease-Classification\ML_Model\plant_disease_prediction_model.h5"
)


# Create your views here.
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype("float32") / 255.0
    return img_array


def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name

@api_view(['POST'])
def make_prediction(request: HttpRequest) -> Response:
    if request.method != 'POST':
        return Response({
            'message': f'Allowed method is POST but got {request.method}.'
        })
    if 'image' not in request.FILES:
        return Response({'error': 'No image file uploaded'}, status=400)

    # Save the image temporarily
    image_file = request.FILES['image']
    temp_image_name = f"{uuid.uuid4()}.jpg"  # Unique filename to avoid conflicts
    temp_image_path = os.path.join('temp_images', temp_image_name)  # Temporary directory

    # Ensure temp directory exists
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)
    
    # Save the file locally
    path = default_storage.save(temp_image_path, ContentFile(image_file.read()))
    class_indices = json.load(r'C:\Users\Valmik Belgaonkar\OneDrive\Desktop\GIS-Plant-Disease-Classification\ML_Model\class_indices.json')
    
    try:
        predicted_class_name = predict_image_class(model=ml_model, image_path=path, class_indices=class_indices)
        return Response({
            'prediction': predicted_class_name
        })
    except Exception as e:
        return Response({'error': f'{e}'}, status=400)
    finally:
            # Clean up: Delete the temporary image after prediction
            if os.path.exists(path):
                os.remove(path)

def home(request: HttpRequest) -> HttpResponse:
    return HttpResponse('<h1>HELLO!!</h1>')