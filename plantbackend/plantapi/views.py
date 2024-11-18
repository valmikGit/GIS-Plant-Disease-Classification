from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from django.http import JsonResponse
from PIL import Image
import numpy as np
import traceback
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

seedling_model = load_model(
    r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\GIS-Plant-Disease-Classification\ML_Model\plant-seedlings-classification\model_best.keras"
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
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


def color_segment_function(img_array):
    img_array = np.rint(img_array)
    img_array = img_array.astype("uint8")
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, (24, 50, 0), (55, 255, 255))
    result = cv2.bitwise_and(img_array, img_array, mask=mask)
    result = result.astype("float64")
    return result


@api_view(["POST"])
def make_prediction_for_seedling(request: HttpRequest) -> Response:
    if request.method != "POST":
        return Response(
            {"message": f"Allowed method is POST but got {request.method}."}
        )
    if "image" not in request.FILES:
        return Response({"error": "No image file uploaded"}, status=400)

    print("Hello")
    print(request.FILES)

    # Save the image temporarily
    try:
        image_file = request.FILES["image"]
        temp_image_name = f"{uuid.uuid4()}.jpg"  # Unique filename to avoid conflicts
        temp_image_path = os.path.join(
            "temp_images", temp_image_name
        )  # Temporary directory
    except Exception as e:
        return JsonResponse({"message": f"Error during file handling: {e}"})

    print("Hello 2")

    # Ensure temp directory exists
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)

    # Save the file locally
    path = default_storage.save(temp_image_path, ContentFile(image_file.read()))

    try:
        # Load an image in RGB format
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        image_rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # Convert to RGB if read in BGR format

        # Apply the color segmentation function
        segmented_image = color_segment_function(image_rgb)

        # Add batch dimension to the segmented image
        segmented_image = np.expand_dims(segmented_image, axis=0)

        # Run the prediction
        predictions = seedling_model.predict(segmented_image)

        # Assuming you already have class weights calculated
        class_weight = {
            0: 1.30,
            1: 0.88,
            2: 1.19,
            3: 0.56,
            4: 1.55,
            5: 0.72,
            6: 0.52,
            7: 1.55,
            8: 0.66,
            9: 1.48,
            10: 0.69,
            11: 0.89,
        }

        # Define your class labels for readability (you may already have this in `label_map`)
        class_labels = {
            0: "Black-grass",
            1: "Charlock",
            2: "Cleavers",
            3: "Common Chickweed",
            4: "Common wheat",
            5: "Fat Hen",
            6: "Loose Silky-hent",
            7: "Maize",
            8: "Scentless Mayweed",
            9: "Shepherds Purse",
            10: "Small-flowered Cranesbill",
            11: "Sugarbeet",
        }

        # Step 1: Apply class weights to the model's predictions
        weighted_predictions = [
            pred * class_weight[i] for i, pred in enumerate(predictions[0])
        ]

        # Step 2 (Optional): Re-normalize to get a probability distribution that sums to 1
        weighted_sum = sum(weighted_predictions)
        normalized_weighted_predictions = [
            wp / weighted_sum for wp in weighted_predictions
        ]

        print('Want to print normalized weights.')
        print(normalized_weighted_predictions)

        # Step 3: Find the class with the highest weighted probability
        predicted_index = np.argmax(normalized_weighted_predictions)
        print(f'Predicted index = {predicted_index}')
        predicted_label = class_labels[predicted_index]
        print(f'Prediction label = {predicted_label}')

        return Response({"prediction": predicted_label})

    except Exception as e:
        # Capture full traceback for debugging
        error_message = traceback.format_exc()
        print(f"Error during prediction: {error_message}")
        return Response({"error": f"An error occurred: {str(e)}"}, status=400)

    finally:
        # Clean up: Delete the temporary image after prediction
        if os.path.exists(path):
            os.remove(path)

# @api_view(["POST"])
# def make_prediction_for_seedling(request: HttpRequest) -> Response:
#     if request.method != "POST":
#         return Response(
#             {"message": f"Allowed method is POST but got {request.method}."}
#         )
#     if "image" not in request.FILES:
#         return Response({"error": "No image file uploaded"}, status=400)

#     print("Received request")
#     print(request.FILES)

#     # Save the image temporarily
#     try:
#         image_file = request.FILES["image"]
#         temp_image_name = f"{uuid.uuid4()}.jpg"  # Unique filename to avoid conflicts
#         temp_image_path = os.path.join(
#             "temp_images", temp_image_name
#         )  # Temporary directory
#     except Exception as e:
#         return JsonResponse({"message": f"Error during file handling: {e}"})

#     print("Processing image...")

#     # Ensure temp directory exists
#     os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)

#     # Save the file locally
#     path = default_storage.save(temp_image_path, ContentFile(image_file.read()))

#     try:
#         # Load the image
#         original_image = cv2.imread(path)
#         original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
#         # Display the original image
#         plt.figure(figsize=(6, 6))
#         plt.imshow(original_image_rgb)
#         plt.title("Original Image")
#         plt.axis("off")
#         plt.show()

#         # Preprocess the image
#         resized_image = cv2.resize(original_image_rgb, (150, 150))

#         # Apply the color segmentation function
#         segmented_image = color_segment_function(resized_image)

#         # Display the preprocessed image
#         plt.figure(figsize=(6, 6))
#         plt.imshow(segmented_image)
#         plt.title("Preprocessed Image")
#         plt.axis("off")
#         plt.show()

#         # Add batch dimension to the segmented image
#         segmented_image = np.expand_dims(segmented_image, axis=0)

#         # Run the prediction
#         predictions = seedling_model.predict(segmented_image)
#         print(f"Raw predictions: {predictions}")

#         # Define your class labels for readability (you may already have this in `label_map`)
#         class_labels = {
#             0: "Black-grass",
#             1: "Charlock",
#             2: "Cleavers",
#             3: "Common Chickweed",
#             4: "Common wheat",
#             5: "Fat Hen",
#             6: "Loose Silky-hent",
#             7: "Maize",
#             8: "Scentless Mayweed",
#             9: "Shepherds Purse",
#             10: "Small-flowered Cranesbill",
#             11: "Sugarbeet",
#         }

#         # Step: Find the class with the highest probability
#         predicted_index = np.argmax(predictions[0])
#         print(f"Predicted index = {predicted_index}")
#         predicted_label = class_labels[predicted_index]
#         print(f"Prediction label = {predicted_label}")

#         return Response({"prediction": predicted_label})

#     except Exception as e:
#         # Capture full traceback for debugging
#         error_message = traceback.format_exc()
#         print(f"Error during prediction: {error_message}")
#         return Response({"error": f"An error occurred: {str(e)}"}, status=400)

#     finally:
#         # Clean up: Delete the temporary image after prediction
#         if os.path.exists(path):
#             os.remove(path)


@api_view(["POST"])
def make_prediction(request: HttpRequest) -> Response:
    if request.method != "POST":
        return Response(
            {"message": f"Allowed method is POST but got {request.method}."}
        )
    if "image" not in request.FILES:
        return Response({"error": "No image file uploaded"}, status=400)

    print("Hello")
    print(request.FILES)

    # Save the image temporarily
    try:
        image_file = request.FILES["image"]
        temp_image_name = f"{uuid.uuid4()}.jpg"  # Unique filename to avoid conflicts
        temp_image_path = os.path.join(
            "temp_images", temp_image_name
        )  # Temporary directory
    except Exception as e:
        return JsonResponse({"message": f"Error during file handling: {e}"})

    print("Hello 2")

    # Ensure temp directory exists
    os.makedirs(os.path.dirname(temp_image_path), exist_ok=True)

    # Save the file locally
    path = default_storage.save(temp_image_path, ContentFile(image_file.read()))

    try:
        # Load class indices for the prediction
        with open(
            r"C:\Users\Valmik Belgaonkar\OneDrive\Desktop\GIS-Plant-Disease-Classification\ML_Model\class_indices.json",
            "r",
        ) as f:
            class_indices = json.load(f)
        print("Class indices loaded.")

        # Run the prediction
        predicted_class_name = predict_image_class(
            model=ml_model, image_path=path, class_indices=class_indices
        )
        print(f"Prediction result: {predicted_class_name}")

        return Response({"prediction": predicted_class_name})

    except Exception as e:
        # Capture full traceback for debugging
        error_message = traceback.format_exc()
        print(f"Error during prediction: {error_message}")
        return Response({"error": f"An error occurred: {str(e)}"}, status=400)

    finally:
        # Clean up: Delete the temporary image after prediction
        if os.path.exists(path):
            os.remove(path)


def home(request: HttpRequest) -> HttpResponse:
    return HttpResponse("<h1>HELLO!!</h1>")
