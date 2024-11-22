import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from math import sqrt, floor
from prettytable import PrettyTable
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Utility Functions
def print_bold(text):
    print('\033[1m{}\033[0m'.format(text))

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        dir_name = '{}{}/'.format(indent, os.path.basename(root))
        if dir_name.strip().startswith('.'):
            continue
        print_bold('\n' + dir_name)
        subindent = ' ' * 4 * (level + 1)
        if level == 0:
            for f in files:
                if f.startswith('.'):
                    continue
                print('{}{}'.format(subindent, f))
        else:
            for i, f in enumerate(files):
                print('{}{}'.format(subindent, f))
                if i == 2:
                    print('{}{}'.format(subindent, '...'))
                    break

def create_validation(validation_split=0.2):
    if os.path.isdir('validation'):
        print('Validation directory already created!')
        return
    os.mkdir('validation')
    for f in os.listdir('train'):
        train_class_path = os.path.join('train', f)
        if os.path.isdir(train_class_path):
            validation_class_path = os.path.join('validation', f)
            os.mkdir(validation_class_path)
            files_to_move = int(validation_split * len(os.listdir(train_class_path)))
            for i in range(files_to_move):
                random_image = os.path.join(train_class_path, random.choice(os.listdir(train_class_path)))
                shutil.move(random_image, validation_class_path)
    print(f'Validation set created successfully using {validation_split:.2%} of training data.')

# Color Segmentation Preprocessing Function
def color_segment_function(img_array):
    img_array = np.rint(img_array).astype('uint8')
    hsv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_img, (24, 50, 0), (55, 255, 255))
    result = cv2.bitwise_and(img_array, img_array, mask=mask).astype('float64')
    return result

# List Dataset Files
list_files(os.getcwd())

# Create Validation Set
create_validation(validation_split=0.2)

# Image Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=color_segment_function
)

test_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=color_segment_function
)

train_generator = train_datagen.flow_from_directory(
    'train',
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

validation_generator = test_datagen.flow_from_directory(
    'validation',
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Class Weights Calculation
label_map = {v: k for k, v in train_generator.class_indices.items()}
class_counts = pd.Series(train_generator.classes).value_counts()
class_weight = {i: 1.0 / c for i, c in class_counts.items()}
norm_factor = np.mean(list(class_weight.values()))
class_weight = {k: v / norm_factor for k, v in class_weight.items()}

# Define Model (Using Pretrained VGG16)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze base layers

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_counts), activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
best_cb = callbacks.ModelCheckpoint(
    'model_best.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto'
)
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
callbacks_list = [best_cb, lr_scheduler, early_stopping]

# Train Model
history = model.fit(
    train_generator,
    class_weight=class_weight,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=callbacks_list
)

# Load Best Model and Test on Raw Images
model = models.load_model('model_best.keras')

for class_name in train_generator.class_indices.keys():
    sample_class = os.path.join('train', class_name)
    random_image = os.path.join(sample_class, random.choice(os.listdir(sample_class)))
    img = image.load_img(random_image, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    predicted_label = label_map[np.argmax(pred)]
    print(f"True label: {class_name}, Predicted label: {predicted_label}")

# Plot Training History
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()
