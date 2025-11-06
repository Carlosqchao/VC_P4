# Práctica 4 y 4b – Detección y lectura de matrículas

## Autores
- **Carlos Ruano Ramos**
- **Juan Boissier García**

---

## Práctica 4 – Detección de matrículas con IA (YOLO)

### Descripción general
En esta práctica se desarrolla un sistema de visión por computadora capaz de detectar matrículas de vehículos en imágenes.  
Para ello, se entrena un modelo de detección de objetos basado en **YOLO (You Only Look Once)**, un algoritmo ampliamente utilizado en tareas de reconocimiento en tiempo real.  
El objetivo es que el modelo aprenda a localizar las matrículas dentro de las imágenes y genere **cajas delimitadoras (bounding boxes)** alrededor de ellas.

---

### Pasos principales

#### 1. Importación de librerías
Se cargan las librerías necesarias para el procesamiento de imágenes, manejo de archivos, entrenamiento del modelo y visualización de resultados.
```python
import cv2  
import math 
import os
import shutil
import random
from pathlib import Path
import yaml
from ultralytics import YOLO
import pandas as pd
import time
import easyocr
import matplotlib.pyplot as plt
```

**Librerías principales:**
- `cv2`: Procesamiento de imágenes y vídeo
- `ultralytics`: Framework para YOLO
- `easyocr` y `pytesseract`: Reconocimiento óptico de caracteres (OCR)
- `pandas`: Gestión de datos y exportación a CSV
- `matplotlib`: Visualización de resultados

#### 2. Preparación del dataset
El dataset se encuentra organizado en:
- Carpetas de `train`, `val` y `test`
- Cada una contiene subcarpetas de `images` y `labels`
- La única clase utilizada es `"matricula"`
- Se genera una distribución aleatoria de las imágenes para evitar sesgos y garantizar una buena generalización del modelo

#### 3. Entrenamiento del modelo YOLO
Se utiliza el framework **Ultralytics YOLO**, en su versión **YOLOv11n**, un modelo ligero y eficiente ideal para entrenamiento en GPU.  
El entrenamiento se realiza utilizando una **GPU NVIDIA RTX 3060** con soporte **CUDA**.

**Comando utilizado para el entrenamiento:**
```bash
yolo detect train model=yolov11n.pt data=data.yaml epochs=100 imgsz=640 device=0 project="C:/Users/Carlos Ruano/Downloads/VC_P4/matriculas_train"
```

**Ejemplo de archivo `data.yaml`:**
```yaml
train: C:/Users/Carlos Ruano/Downloads/VC_P4/TGC_RBNW/train/images
val: C:/Users/Carlos Ruano/Downloads/VC_P4/TGC_RBNW/val/images
nc: 1
names: ['matricula']
```

#### 4. Validación y resultados
- Se evalúa el modelo con el conjunto de validación
- Se muestran ejemplos de detecciones, donde las matrículas se enmarcan con rectángulos
- Ejemplo de detección y vídeo: [Ver ejemplo en YouTube]

---

## Práctica 4b – Detección completa y OCR en vídeo

### Descripción general
Esta extensión de la práctica integra tres componentes principales:
1. **Detección de personas y vehículos** (usando YOLO11n general)
2. **Detección específica de matrículas** (usando el modelo entrenado)
3. **Reconocimiento de texto (OCR)** en las matrículas detectadas

El sistema procesa vídeos fotograma a fotograma, aplicando desenfoque (blur) a personas y matrículas para proteger la privacidad, mientras extrae y registra toda la información relevante.

---

### Arquitectura del sistema

#### Modelos utilizados
```python
# Modelo general para personas y coches
model_general = YOLO("yolo11n.pt")

# Modelo especializado en matrículas
model_matriculas = YOLO("matriculas_train/weights/best.pt")

# Sistemas OCR
reader = easyocr.Reader(['es'], gpu=True)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

---

### Pipeline de procesamiento

#### Configuración inicial
```python
# Rutas y parámetros
video_input = "C0142.MP4"
video_output = "video_resultado_final_blur.mp4"
csv_output = "CSVs/resultados_final_blur.csv"
blur_kernel = (25, 25)
```

#### Estructura del CSV de salida
| Campo | Descripción |
|-------|-------------|
| `frame` | Número de fotograma |
| `id_tracking` | ID de seguimiento del objeto |
| `tipo_objeto` | Tipo: person, car, matricula |
| `confianza` | Confianza de la detección (0-1) |
| `x1, y1, x2, y2` | Coordenadas de la caja delimitadora |
| `texto_matricula` | Texto OCR extraído |

---

### Flujo de procesamiento por fotograma

#### 1. Detección de personas y vehículos
```python
results_general = model_general.track(frame, persist=True, classes=[0, 2])
# 0 = person
# 2 = car
```

**Funcionalidades:**
- Tracking persistente de objetos
- Conteo de objetos únicos
- Aplicación de blur a personas
- Dibujo de cajas delimitadoras con colores diferenciados:
  - Verde: Personas
  - Naranja: Coches

#### 2. Detección de matrículas
```python
results_matriculas = model_matriculas(frame)[0]
```

**Procesamiento de cada matrícula:**
1. Extracción de región de interés (ROI) con padding
2. Preprocesamiento de imagen para OCR
3. Aplicación de OCR (antes del blur)
4. Aplicación de blur gaussiano
5. Guardado de la ROI original

#### 3. Preprocesamiento para OCR
```python
# Aplicar padding
pad = 5
y1_p, y2_p = max(0, y1-pad), min(frame.shape[0], y2+pad)
x1_p, x2_p = max(0, x1-pad), min(frame.shape[1], x2+pad)
roi = frame[y1_p:y2_p, x1_p:x2_p]

# Conversión a escala de grises
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Redimensionado para mejorar OCR
scale = 300 / roi_gray.shape[1]
new_h = int(roi_gray.shape[0] * scale)
roi_gray = cv2.resize(roi_gray, (300, new_h))

# Umbral adaptativo
roi_bin = cv2.adaptiveThreshold(
    roi_gray, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 11, 2
)

# Dilatación para mejorar caracteres
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
roi_bin = cv2.dilate(roi_bin, kernel, iterations=1)
```

#### 4. Extracción de texto con OCR

**EasyOCR:**
```python
ocr_result = reader.readtext(roi_bin)
texto_easy = "".join([r[1] for r in ocr_result])
```

**Tesseract:**
```python
config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
texto_tess = pytesseract.image_to_string(roi_bin, config=config)
texto_tess = texto_tess.strip().replace(" ", "")
```

**Diferencias clave:**
- **EasyOCR**: Basado en deep learning, mejor con imágenes de baja calidad
- **Tesseract**: Motor OCR tradicional, configurable, rápido

#### 5. Visualización y seguimiento
```python
# Contador global en pantalla
cv2.putText(frame, 
    f"Personas: {object_counts['person']} | Coches: {object_counts['car']} | Matrículas: {object_counts['matricula']}",
    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

# Etiquetas en objetos
cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
cv2.putText(frame, etiqueta, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
```

---

### Comparación de motores OCR

#### Script de análisis comparativo
```python
from difflib import SequenceMatcher

# Cargar ambos CSVs
df_easy = pd.read_csv("CSVs/resultados_final_blur.csv")
df_tess = pd.read_csv("CSVs/resultados_final_blur_tesseract.csv")

# Filtrar solo matrículas
df_easy = df_easy[df_easy["tipo_objeto"] == "matricula"]
df_tess = df_tess[df_tess["tipo_objeto"] == "matricula"]

# Emparejar por coordenadas
merged = pd.merge(df_easy, df_tess, 
    on=["frame", "x1", "y1", "x2", "y2"],
    suffixes=("_easy", "_tess"))

# Calcular similitud
def similarity(a, b):
    return SequenceMatcher(None, str(a).strip(), str(b).strip()).ratio()

merged["similitud"] = merged.apply(
    lambda row: similarity(row["texto_matricula_easy"], row["texto_matricula_tess"]),
    axis=1
)
```

#### Métricas de evaluación
```python
promedio_similitud = merged["similitud"].mean() * 100
exactos = (merged["similitud"] == 1.0).sum()
total = len(merged)
exactitud = (exactos / total * 100) if total > 0 else 0

print(f"Total de matrículas comparadas: {total}")
print(f"Coincidencias exactas: {exactos} ({exactitud:.2f}%)")
print(f"Similitud promedio: {promedio_similitud:.2f}%")
```

---

### Procesamiento de imágenes estáticas (Test)

#### Descripción
Sistema para procesar el conjunto de test y generar un CSV con todas las detecciones y lecturas OCR.
```python
# Configuración
model_path = "matriculas_train/weights/best.pt"
test_dir = "TGC_RBNW/test/images"
csv_output = "CSVs/matriculas_test.csv"

# Procesar todas las imágenes
for filename in os.listdir(test_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue
    
    image_path = os.path.join(test_dir, filename)
    frame = cv2.imread(image_path)
    results = model(frame)[0]
    
    # Detección y OCR
    for box in results.boxes:
        # ... (similar al procesamiento de vídeo)
```

---

### Visualización interactiva

#### Script de prueba con matplotlib
```python
# Configuración
model_path = "matriculas_train/weights/best.pt"
image_paths = ["test.jpeg", "test2.jpeg"]

model = YOLO(model_path)
reader = easyocr.Reader(['es'], gpu=True)

for image_path in image_paths:
    frame = cv2.imread(image_path)
    results = model(frame)[0]
    
    # Procesar y visualizar
    for box in results.boxes:
        # ... OCR y anotaciones
    
    # Mostrar con matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.title(image_path)
    plt.imshow(frame_rgb)
    plt.axis("off")
    plt.show()
```

---

### Optimizaciones y consideraciones técnicas

#### 1. Gestión de memoria
- Uso de `pd.concat()` con `ignore_index=True` para construcción eficiente de DataFrames
- Liberación de recursos con `cap.release()`, `out.release()`, `cv2.destroyAllWindows()`

#### 2. Preprocesamiento adaptativo
- Umbral adaptativo de Gauss en lugar de umbral fijo
- Redimensionado proporcional a 300px de ancho
- Dilatación morfológica para mejorar la continuidad de caracteres

#### 3. Gestión de errores
- Validación de tamaño de ROI antes de OCR
- Control de límites en operaciones de padding
- Verificación de imágenes válidas antes de procesamiento

#### 4. Rendimiento
```python
# Medición de tiempos
tiempos = []
start_time = time.time()
# ... procesamiento ...
tiempos.append(time.time() - start_time)

# Reporte final
print(f"Tiempo promedio de inferencia: {sum(tiempos)/len(tiempos):.4f}s/frame")
```

---

### Estructura de salida
```
proyecto/
├── video_resultado_final_blur.mp4      # Vídeo procesado con blur
├── CSVs/
│   ├── resultados_final_blur.csv       # Resultados con EasyOCR
│   ├── resultados_final_blur_tesseract.csv  # Resultados con Tesseract
│   ├── comparacion_easy_tesseract.csv  # Análisis comparativo
│   └── matriculas_test.csv             # Resultados del conjunto test
└── matriculas_detectadas/              # ROIs originales guardadas
    ├── frame_1_x123_y456.jpg
    ├── frame_2_x789_y012.jpg
    └── ...
```
#### Ejemplos de salida
<img width="366" height="504" alt="imagen" src="https://github.com/user-attachments/assets/287bff85-4236-47c1-b8ca-306dbd075d22" />
<img width="366" height="504" alt="imagen" src="https://github.com/user-attachments/assets/fc3ce242-54c5-47f6-a750-82e92c33a55b" />

### Video

[![Ver video en YouTube](https://img.youtube.com/vi/RArtErWv0_I/hqdefault.jpg)](https://youtu.be/RArtErWv0_I)

### Comparación Visual de OCRs

<img width="854" height="473" alt="image" src="https://github.com/user-attachments/assets/01ca871f-f2c1-439e-b7f2-4488751b5001" />
<img width="701" height="451" alt="image" src="https://github.com/user-attachments/assets/32282887-3282-46f5-a11f-feec74d0b015" />
<img width="541" height="449" alt="image" src="https://github.com/user-attachments/assets/f47be625-e727-4bf3-8452-27ccd0f50c7c" />




## Referencias
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [EasyOCR GitHub](https://github.com/JaidedAI/EasyOCR)
- [Tesseract OCR Documentation](https://github.com/tesseract-ocr/tesseract)
- [OpenCV Documentation](https://docs.opencv.org/)
