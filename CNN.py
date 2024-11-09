# Importación de Librerías
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Directorios de datos
train_dir = ' '
test_dir = 'C:/Users/PERSONAL/Downloads/uco-animals-vs-plants/test'

# Configuración de parámetros
batch_size = 32
img_height, img_width = 224, 224
epochs = 30

# =============================================================================
# Parte 1: Red Binaria para Clasificar entre Animales y Flores
# =============================================================================

# Generadores de imágenes para la red binaria
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',  # Binario: animales vs flores
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# Definición de la red binaria
binary_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Salida binaria
])

# Compilación y entrenamiento de la red binaria
binary_model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

binary_model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# =============================================================================
# Parte 2: Redes para Clasificar Animales y Flores
# =============================================================================

# -------------------------
# Red para Clasificar Animales (4 clases)
# -------------------------

# Generadores de imágenes solo para animales
animal_train_dir = os.path.join(train_dir, 'animals')
animal_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    shear_range=0.2,
    brightness_range=[0.8, 1.2]
)

animal_train_generator = animal_datagen.flow_from_directory(
    animal_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

animal_validation_generator = animal_datagen.flow_from_directory(
    animal_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Definición de la red para animales
animal_model = Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(animal_train_generator.num_classes, activation='softmax')
])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_animal_model.h5', monitor='val_accuracy', save_best_only=True)

# Compilación y entrenamiento de la red para animales
animal_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

animal_model.fit(
    animal_train_generator,
    steps_per_epoch=animal_train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=animal_validation_generator,
    validation_steps=animal_validation_generator.samples // batch_size,
    callbacks=[early_stopping, checkpoint]
)

# -------------------------
# Red para Clasificar Flores (12 clases)
# -------------------------

# Generadores de imágenes solo para flores
flower_train_dir = os.path.join(train_dir, 'flowers')
flower_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

flower_train_generator = flower_datagen.flow_from_directory(
    flower_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

flower_validation_generator = flower_datagen.flow_from_directory(
    flower_train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Definición de la red para flores
flower_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(flower_train_generator.num_classes, activation='softmax')
])

# Compilación y entrenamiento de la red para flores
flower_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

flower_model.fit(
    flower_train_generator,
    steps_per_epoch=flower_train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=flower_validation_generator,
    validation_steps=flower_validation_generator.samples // batch_size
)

# =============================================================================
# Parte 3: Predicción en el Conjunto de Prueba
# =============================================================================

# Cargar las imágenes de prueba y hacer predicciones en dos etapas
predictions = []

for filename in os.listdir(test_dir):
    img_path = os.path.join(test_dir, filename)
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Paso 1: Determinar si es animal o flor
    binary_prediction = binary_model.predict(img_array)
    is_flower = binary_prediction[0][0] > 0.5

    if is_flower:
        # Paso 2a: Clasificar como flor
        flower_prediction = flower_model.predict(img_array)
        predicted_class = np.argmax(flower_prediction[0])
        label = list(flower_train_generator.class_indices.keys())[predicted_class]

    else:
        # Paso 2b: Clasificar como animal
        animal_prediction = animal_model.predict(img_array)
        predicted_class = np.argmax(animal_prediction[0])
        label = list(animal_train_generator.class_indices.keys())[predicted_class]

    predictions.append([filename, label])

# Guardar las predicciones en un archivo CSV
predictions_df = pd.DataFrame(predictions, columns=['file', 'label'])
predictions_df.to_csv('C:/Users/PERSONAL/Downloads/uco-animals-vs-plants/predictions.csv', index=False)

print("Predicciones completadas y guardadas en 'predictions.csv'")
