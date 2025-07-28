# 🤖 Reconocimiento de Gestos con IA  

🔗 **[Ver demo en GitHub Pages](https://luchoaguero44.github.io/Lengua-de-Senhas/)**

Este proyecto utiliza inteligencia artificial para el reconocimiento de gestos a través de un modelo basado en redes neuronales convolucionales (CNN). Su objetivo es identificar distintos gestos realizados con las manos y convertirlos en información útil para facilitar la comunicación.  

## ¿Cómo funciona?  
El proyecto se divide en varias etapas que trabajan juntas para lograr el reconocimiento de gestos:  

1. **Seguimiento de manos (SeguimientoManos.py):**  
   Este archivo contiene funciones clave para detectar y seguir las manos en tiempo real. Utiliza técnicas avanzadas de visión por computadora para identificar los puntos clave de la mano y crear un cuadro delimitador alrededor de ella.  

2. **Creación del dataset (Data.py):**  
   Usando `SeguimientoManos.py`, este archivo genera un dataset de imágenes de gestos. Identifica las manos, marca los puntos clave y guarda imágenes recortadas de los gestos en carpetas específicas. Este dataset es fundamental para entrenar el modelo.  

3. **Entrenamiento del modelo (Train.py):**  
   Este archivo entrena una red neuronal convolucional (CNN) utilizando el dataset generado. El modelo aprende a identificar y clasificar los gestos según las imágenes procesadas.  

4. **Pruebas del modelo (Prueba.py):**  
   Aquí se implementa el modelo entrenado para realizar el reconocimiento de gestos en tiempo real. Al procesar un video o una imagen, el sistema detecta los gestos y los clasifica correctamente.  

## Estructura del Proyecto  

(📁 Aquí puedes agregar un esquema de carpetas si lo deseas)

## Requisitos  
- Python 3.x  
- Librerías: `opencv-python`, `numpy`, `tensorflow`, `mediapipe`, `streamlit`  

## Cómo usar  
1. **Generar dataset:**  
   Ejecuta `Data.py` para crear un conjunto de datos de imágenes de los gestos que deseas reconocer. Primeramente creando una carpeta `Data` donde se almacenarán las imágenes.

2. **Entrenar el modelo:**  
   Usa `Train.py` para entrenar el modelo con el dataset generado.  

3. **Reconocimiento de gestos:**  
   Ejecuta `Prueba.py` para probar el modelo entrenado en tiempo real o con imágenes cargadas.  
