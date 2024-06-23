# Proyecto MNIST
El proyecto MNIST es un ejemplo clásico en el campo del aprendizaje automático y la visión por computadora. Consiste en clasificar imágenes de dígitos escritos a mano en sus respectivas etiquetas numéricas. El conjunto de datos MNIST (Instituto Nacional de Normas y Tecnología Modificado) es una gran base de datos de dígitos manuscritos que se utiliza habitualmente para entrenar diversos sistemas de procesamiento de imágenes y modelos de aprendizaje automático. Se creó "remezclando" las muestras de los conjuntos de datos originales del NIST y se ha convertido en una referencia para evaluar el rendimiento de los algoritmos de clasificación de imágenes.

## Características
- MNIST contiene 60.000 imágenes de entrenamiento y 10.000 imágenes de prueba de dígitos manuscritos.
- El conjunto de datos consta de imágenes en escala de grises de tamaño 28x28 píxeles.
- Las imágenes se normalizan para que quepan en un cuadro delimitador de 28x28 píxeles y se suavizan, introduciendo niveles de escala de grises.

## Estructura del conjunto de datos
El conjunto de datos MNIST se divide en dos subconjuntos:
1. **Conjunto de entrenamiento**: Este subconjunto contiene 60.000 imágenes de dígitos manuscritos utilizadas para entrenar modelos de aprendizaje automático.
2. **Conjunto de pruebas**: Este subconjunto consta de 10.000 imágenes utilizadas para probar y evaluar los modelos entrenados.

<figure>
    <img src="/data/mnist_digits.png"
         alt="hand-written digits">
    <figcaption>Ejemplo de variedad y complejidad de los dígitos manuscritos del conjunto de datos MNIST.</figcaption>
</figure>

## Instalación de paquetes
Los paquetes necessarios se encuentran listados en el archivo *requirements.txt*.
- Crear el entorno virtual: `python3 -m venv venv`
- Activar el entorno virtual: `source venv/bin/activate`
- Upgrade pip: `python3 -m pip install --upgrade pip`
- Instalar paquetes desde *requirements.txt*: `pip install -r requirements.txt`
- Desinstalar todos los paquetes: `pip freeze | xargs pip uninstall -y`
- Desactivar el entorno virtual: `deactivate`
- Eliminar le entorno virtual (opcional): `rm -rf venv`

### CEIA - FIUBA
Este repositorio forma parte del proyecto final de la materia de Aprendizaje de Máquina 1. Materia que se dicta en la Carrera de Especialización en Inteligencia Artificial, en la Facultad de Ingeniería de la Universidad de Buenos Aires. El proyecto trata con el dataset MNNIST y la implementación de redes neuronales para el entrenamiento del modelo. El lector debe referirse a las *notebooks* en jupyter (.ipynb) tales como: *analysis*, *model* y *augmentation* para el anaálisis estadístico, el entrenamiento del modelo y el proceso de *data augmentation*, respectivamente. Luego, dentro de la carpeta `src/` se encuentra el código fuente o desarrollos propios, lo cual se invita a explorar en caso de querer entender lo que se realizó en los jupiter notebooks.
