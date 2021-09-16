# Detección de Objetos Abandonados en Tiempo Real y Entornos Controlados a través de Redes Neuronales Convolucionales
### Tesis para el Máster en Informática - Universidad de Málaga.
##### La detección de objetos abandonados en sistemas inteligentes de visión por computadora tiene como objetivo primordial evitar catástrofes, estos sistemas deben ser precisos y de bajo costo computacional. En esta memoria se expone un modelo para detección de objetos abandonados a través de la técnica background subtraction que integra YOLOv5s sistema de detección de objetos en tiempo real de última generación. Utilizado en este modelo de clasificar los objetos desatendidos o abandonados. 
##### Para la evaluación del rendimiento se utilizan los conjuntos de datos ABODA y AVSS 2007 y secuencias de videos en tiempo real. El conjunto de datos para capacitación de YOLOv5s está compuesto por 457 imágenes categorizadas como maletas de viaje, morrales y paquetes.
##### El trabajo ha sido desarrollado bajo el lenguaje de programación Python haciendo uso de las librerías de Opencv y PyTorch v1.7.1.
### Requerimientos
##### Necesita tener una GPU, Windows instalado, python 3.8.
### Entrenamiento de YOLOv5
##### Ejecutar el archivo TRAIN_YOLO_V5_ON_YOUR_CUSTOM_DATASET.ipynb
#### Descargar el modelo Entrenado directorio yolov5
### Ejecución
##### Una vez entrenado el modelo YOLOv5s en el archivo TFMV2.py
##### modificar la línea model = torch.hub.load('ultralytics/yolov5', 'yolov5x') #carga del modelo preeentrenado desde ultralytics 
##### por model =torch.hub.load('yolov5', 'custom', path='yolov5//runs//train//exp2//weights//best.pt', source='local') 
#### ejecutar el archivo TFMV2.py
### Notas: 
##### los videos NE1.mp4 y NE1.mp4 están grabados a una resolución de 1280x720 por lo que es necesario activar en el archivo TFMV2.py la línea
##### frame=cv2.resize(frame,(1280,720),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
