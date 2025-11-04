# Práctica 4 y 4b – Detección y lectura de matrículas
## Autores: 
- Carlos Ruano Ramos
- Juan Boissier García

## Práctica 4 – Detección de matrículas con IA (YOLO)

### Descripción general
En esta práctica se desarrolla un sistema de visión por computadora capaz de detectar matrículas de vehículos en imágenes.  
Para ello, se entrena un modelo de detección de objetos basado en **YOLO (You Only Look Once)**, un algoritmo ampliamente utilizado en tareas de reconocimiento en tiempo real.  

El objetivo es que el modelo aprenda a localizar las matrículas dentro de las imágenes y genere **cajas delimitadoras (bounding boxes)** alrededor de ellas.

---

### Pasos principales

1. **Importación de librerías**  
   Se cargan las librerías necesarias para el procesamiento de imágenes, manejo de archivos, entrenamiento del modelo y visualización de resultados.  
   Entre ellas se encuentran: `cv2`, `os`, `math`, `random`, `yaml`, `ultralytics`, `pandas` y `matplotlib`.

2. **Preparación del dataset**  

   El dataset se encuentra en: https://drive.google.com/file/d/1zkZfTeM3Q0ETdCEkMXRJCbN5KiwKV6am/view?usp=sharing

   - El conjunto de imágenes se divide en carpetas de `train`, `val` y `test`.  
   - Cada una contiene subcarpetas de `images` y `labels`.  
   - La única clase utilizada es `"matricula"`.  
   - Se genera una distribución aleatoria de las imágenes para evitar sesgos y garantizar una buena generalización del modelo.


3. **Entrenamiento del modelo YOLO**  
   - Se utiliza el framework **Ultralytics YOLO**, en su versión **YOLOv11n**, un modelo ligero y eficiente ideal para entrenamiento en GPU.  
   - El entrenamiento se realiza utilizando una **GPU NVIDIA RTX 3060** con soporte **CUDA**.  
   - El comando utilizado para entrenar el modelo sobre el dataset fue el siguiente:

```bash
yolo detect train model=yolov11n.pt data=data.yaml epochs=100 imgsz=640 device=0
```

4. **Validación y resultados**  
   - Se evalúa el modelo con el conjunto de validación.  
   - Se muestran ejemplos de detecciones, donde las matrículas se enmarcan con rectángulos.

#### Ejemplo de detección y link del video:

[![Video](https://github.com/user-attachments/assets/81873980-6e1a-4ee9-ab46-e053c7db19b9)](https://youtu.be/RArtErWv0_I)


---

## Práctica 4b – Lectura de matrículas mediante OCR

### Descripción general
En esta práctica se amplía el trabajo anterior utilizando el modelo entrenado con YOLO para detectar las matrículas y posteriormente aplicar un sistema de **Reconocimiento Óptico de Caracteres (OCR)**, utilizando la librería **EasyOCR**, con el fin de **leer el texto** presente en cada matrícula detectada.  

El resultado final es un sistema que no solo localiza las matrículas en las imágenes, sino que también **extrae los caracteres alfanuméricos** contenidos en ellas.

---

### Pasos principales

1. **Carga del modelo YOLO entrenado**  
   Se carga el modelo generado en la Práctica 4 para realizar detecciones en nuevas imágenes.

2. **Detección y recorte de matrículas**  
   - El modelo detecta las matrículas en las imágenes de entrada.  
   - Se recortan las regiones correspondientes a las matrículas detectadas para analizarlas individualmente.

3. **Lectura del texto con EasyOCR**  
   - Se aplica la librería **EasyOCR** para reconocer los caracteres en cada recorte.  
   - El resultado se muestra en consola o sobre las imágenes originales, indicando el texto reconocido.

4. **Visualización de resultados**  
   - Se presentan las imágenes con las cajas delimitadoras y el texto reconocido superpuesto.  
   - Esto permite verificar la precisión del reconocimiento y la correcta segmentación de las matrículas.

#### Ejemplo de lectura de matrícula:

<img width="470" height="676" alt="image" src="https://github.com/user-attachments/assets/da245551-2683-41f4-a99a-c2669e8128ea" />


---

## Requisitos del entorno

- Python 3.8 o superior  
- Librerías necesarias:

```bash
pip install ultralytics easyocr opencv-python matplotlib pandas pyyaml
```

---

## Estructura sugerida del proyecto

```
VC_P4/
├── datasets/
│   ├── images/
│   ├── labels/
├── runs/
│   └── detect/
├── VC_P4.ipynb
└── README_Practica4.md
```

---

## Notas finales
- Esta práctica permite comprender el flujo completo de un sistema de detección y reconocimiento: **desde el entrenamiento del modelo de detección** hasta **la interpretación del texto en las imágenes**.  
- Se recomienda ajustar los parámetros de entrenamiento y las configuraciones del OCR según la calidad del dataset para mejorar los resultados.
