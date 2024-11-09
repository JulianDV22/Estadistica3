import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import glob
import cv2
import os

# Configuración inicial
img_height, img_width = 64, 64
label_map = {label: idx for idx, label in enumerate([
    "Black-grass", "cat", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "cow", "dog", "Fat Hen", "horse",
    "Loose Silky-bent", "Maize", "Scentless Mayweed", "Shepherds Purse",
    "Small-flowered Cranesbill", "Sugar beet"
])}
num_classes = len(label_map)


# Carga y preprocesamiento de datos
def load_data(path, is_test=False):
    images = []
    labels = []
    if is_test:
        for img_path in glob.glob(f"{path}/*.png"):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"No se pudo cargar la imagen: {img_path}")
                continue
            img = cv2.resize(img, (img_width, img_height))
            images.append((img.flatten() / 255.0).astype(np.float32))
        return np.array(images)
    else:
        for label in os.listdir(path):
            label_folder = os.path.join(path, label)
            if os.path.isdir(label_folder):
                for img_path in glob.glob(f"{label_folder}/*.png"):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"No se pudo cargar la imagen: {img_path}")
                        continue
                    img = cv2.resize(img, (img_width, img_height))
                    images.append((img.flatten() / 255.0).astype(np.float32))
                    labels.append(label_map[label])
        return np.array(images), np.array(labels)


# Cargar datos de entrenamiento
X_train, y_train = load_data("C:/Users/PERSONAL/Downloads/uco-animals-vs-plants/train")
X_test = load_data("C:/Users/PERSONAL/Downloads/uco-animals-vs-plants/test", is_test=True)

# Dividir los datos de entrenamiento para validación
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convertir los datos a tensores de PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)


# Modelo de Regresión Logística usando PyTorch
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


# Crear el modelo
input_size = img_height * img_width
model = LogisticRegressionModel(input_size, num_classes)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Entrenamiento del modelo
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Época [{epoch + 1}/{num_epochs}], Pérdida: {loss.item():.4f}')

# Validación del modelo
with torch.no_grad():
    val_outputs = model(X_val_tensor)
    _, val_predicted = torch.max(val_outputs, 1)
    val_accuracy = accuracy_score(y_val_tensor, val_predicted)
    print(f"Precisión en el conjunto de validación: {val_accuracy:.4f}")

# Predicciones en el conjunto de prueba
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, y_pred = torch.max(test_outputs, 1)

# Convertir los índices de predicción a etiquetas de clase
y_pred_labels = [list(label_map.keys())[list(label_map.values()).index(pred)] for pred in y_pred.numpy()]

# Generación de archivo de predicciones
image_files = sorted(glob.glob("C:/Users/PERSONAL/Downloads/uco-animals-vs-plants/test/*.png"))
image_names = [os.path.basename(img) for img in image_files]
submission_logistic = pd.DataFrame({"file": image_names, "label": y_pred_labels})
submission_logistic.to_csv("C:/Users/PERSONAL/Downloads/uco-animals-vs-plants/submission_logistic.csv", index=False)

print("Archivo de predicciones 'submission_logistic_pytorch.csv' generado correctamente.")

