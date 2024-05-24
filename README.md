<!-- Code -->
Los paquetes necessarios se encuentran listados en el archivo *requirements.txt*. Para instalarlos se deben realizar previamente los siguientes pasos:
1. Crear el entorno virtual:
`python3 -m venv venv`
2. Activar el entorno virtual:
`source venv/bin/activate`
3. Instalar paquetes desde *requirements.txt*: 
`pip install -r requirements.txt`
4. Desinstalar todos los paquetes:
`pip freeze | xargs pip uninstall -y`
5. Desactivar el entorno virtual:
`deactivate`
6. Eliminar le entorno virtual (opcional):
`rm -rf venv`

## Proyecto MNIST

El proyecto MNIST es un ejemplo clásico en el campo del aprendizaje automático y la visión por computadora. Consiste en clasificar imágenes de dígitos escritos a mano en sus respectivas etiquetas numéricas.

### Estructura del proyecto

El proyecto MNIST se organiza de la siguiente manera:
- naivenn/
    - data/
      - mnist.pkl.gz
      - data_from_result_01.txt
      - ...
      - data_from_result_xx.txt
    - run/
      - noreg_varhl.py
      - noreg_varlr.py
      - ...
    - src/
      - __init__.py
      - mnist.py
      - nnnumpy.py
      - nntorch.py
      - nn.py
      - scalar.py
      - vector.py
    - venv/
    - Dockerfile
    - README.md
    - requirements.txt
