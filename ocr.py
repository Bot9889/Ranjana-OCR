import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models


data_dir = 'Dataset/'
img_height, img_width = 64, 64
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)



model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# 
model.fit(train_data, validation_data=val_data, epochs=10)
model.save('ranjana_ocr_model.h5')

# 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import ImageFont, ImageDraw, Image as PILImage

# Load model and class labels
model = load_model('ranjana_ocr_model.h5')
labels = {v: k for k, v in train_data.class_indices.items()}

# Load and preprocess test image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width), color_mode='grayscale')
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Predict and render
def predict_and_render(img_path, font_path='Ranjana Regular.ttf', output_path='output.png'):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    predicted_label = labels[class_index]
    print("Predicted Label:", predicted_label)

    # Render using Ranjana font
    img = PILImage.new("RGB", (300, 100), color="white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 72)
    draw.text((10, 10), predicted_label, font=font, fill="black")
    
    # Save the image
    img.save(output_path)
    print(f"Output saved to {output_path}")

# Example usage
predict_and_render('test_character.png')
