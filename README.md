# Transfer Learning para Clasificación de Frutas y Verduras

Proyecto de Transfer Learning utilizando MobileNetV3 para clasificar 5 tipos de frutas y verduras: manzana, granada, mango, limón y naranja.

## Descripción del Proyecto

Este proyecto implementa dos variantes de clasificadores de imágenes basados en Transfer Learning con MobileNetV3:

- **Versión 1 (Simple)**: Clasificador básico sin Batch Normalization ni Dropout
- **Versión 2 (Extendido)**: Clasificador con arquitectura tipo embudo, incluyendo Batch Normalization y Dropout con probabilidades variables

El objetivo es comparar el desempeño de ambas variantes en la clasificación de imágenes de frutas y verduras, evaluando el impacto de las técnicas de regularización.

## Estructura del Proyecto

```
transfer-learning-frutas-verduras-pytorch/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── config.py           # Configuración centralizada del proyecto
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py      # Carga de datos y transformaciones
│   ├── models/
│   │   ├── __init__.py
│   │   └── mobilenet_classifier.py  # Implementación de modelos
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Lógica de entrenamiento
│   │   └── early_stopping.py   # Early stopping
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py        # Evaluación de modelos
│   │   └── metrics.py          # Métricas y visualizaciones
│   └── utils/
│       ├── __init__.py
│       └── visualization.py    # Utilidades de visualización
├── app/
│   └── app.py                  # Aplicación Streamlit
├── notebooks/
│   └── transfer_learning_notebook.ipynb  # Notebook principal
├── saved_models/               # Modelos entrenados
├── Datos/                      # Dataset (no versionado)
│   ├── train/
│   ├── validation/
│   └── test/
├── train.py                    # Script de entrenamiento
├── evaluate.py                 # Script de evaluación
├── requirements.txt            # Dependencias del proyecto
├── .gitignore
└── README.md
```

## Clases Seleccionadas

El proyecto trabaja con las siguientes 5 clases:

1. Apple (Manzana)
2. Pomegranate (Granada)
3. Mango
4. Lemon (Limón)
5. Orange (Naranja)

## Características Técnicas

### Modelo Base
- **Arquitectura**: MobileNetV3 Large
- **Pesos preentrenados**: ImageNet
- **Técnica**: Transfer Learning

### Variantes del Clasificador

#### Versión 1 - Simple
- Una capa Fully Connected
- Sin Batch Normalization
- Sin Dropout
- Activación: Ninguna (logits)

#### Versión 2 - Extendido
- Arquitectura tipo embudo: 512 → 256 → 128 → 5
- Batch Normalization después de cada capa lineal
- Dropout con probabilidades incrementales (0.2 a 0.5)
- Activación: ReLU

### Data Augmentation
- Redimensionado y recorte aleatorio
- Rotación aleatoria (±30°)
- Flip horizontal y vertical
- Ajustes de brillo, contraste y saturación
- Normalización con estadísticas de ImageNet

### Técnicas de Regularización
- Early Stopping (paciencia: 10 epochs)
- Weight Decay
- Dropout (solo en Versión 2)
- Learning Rate Scheduler

## Instalación

### Requisitos Previos
- Python 3.8 o superior
- CUDA (opcional, para GPU)

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Dependencias Principales
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- streamlit >= 1.28.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0

## Uso

### 1. Entrenamiento desde la Línea de Comandos

#### Entrenar Versión 1 (Simple)
```bash
python train.py --variant simple --epochs 50 --batch-size 32 --lr 0.001
```

#### Entrenar Versión 2 (Extendido)
```bash
python train.py --variant extended --epochs 50 --batch-size 32 --lr 0.001
```

#### Opciones Disponibles
- `--variant`: Variante del modelo (`simple` o `extended`)
- `--epochs`: Número de epochs de entrenamiento
- `--batch-size`: Tamaño del batch
- `--lr`: Learning rate
- `--device`: Dispositivo (`cuda` o `cpu`)
- `--no-early-stopping`: Deshabilitar early stopping

### 2. Evaluación de Modelos

```bash
python evaluate.py --model-path saved_models/mobilenet_v3_simple_best.pth --variant simple --dataset test
```

### 3. Uso del Notebook

Abre el notebook en Jupyter:

```bash
jupyter notebook notebooks/transfer_learning_notebook.ipynb
```

El notebook incluye:
- Carga y exploración de datos
- Entrenamiento de ambas variantes
- Evaluación y comparación
- Visualizaciones

### 4. Interfaz Gráfica con Streamlit

```bash
streamlit run app/app.py
```

La aplicación permite:
- Cargar imágenes desde archivo o cámara
- Seleccionar variante del modelo
- Visualizar predicciones con probabilidades
- Interfaz intuitiva y responsive

## Configuración

Todos los parámetros del proyecto se centralizan en `src/config/config.py`:

```python
# Clases seleccionadas
SELECTED_CLASSES = ['apple', 'pomegranate', 'mango', 'lemon', 'orange']

# Hiperparámetros
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10

# Data Augmentation
AUGMENTATION = {
    'rotation': 30,
    'horizontal_flip': 0.5,
    'vertical_flip': 0.2,
    'brightness': 0.2,
    'contrast': 0.2,
    'saturation': 0.2,
}
```

## Resultados Esperados

El entrenamiento genera:

1. **Modelos guardados** (`.pth`) en `saved_models/`
2. **Curvas de aprendizaje** (gráficos de loss y accuracy)
3. **Matriz de confusión** normalizada
4. **Métricas por clase**: Precision, Recall, F1-Score
5. **Reporte de clasificación** completo

## Métricas de Evaluación

- **Accuracy**: Exactitud global del modelo
- **Precision**: Proporción de predicciones positivas correctas
- **Recall**: Proporción de positivos reales identificados
- **F1-Score**: Media armónica de Precision y Recall
- **Matriz de Confusión**: Visualización de predicciones vs etiquetas reales

## Estructura de Datos

El dataset debe organizarse de la siguiente manera:

```
Datos/
├── train/
│   ├── apple/
│   ├── pomegranate/
│   ├── mango/
│   ├── lemon/
│   └── orange/
├── validation/
│   ├── apple/
│   ├── pomegranate/
│   ├── mango/
│   ├── lemon/
│   └── orange/
└── test/
    ├── apple/
    ├── pomegranate/
    ├── mango/
    ├── lemon/
    └── orange/
```

## Tecnologías Utilizadas

- **Deep Learning**: PyTorch, torchvision
- **Visualización**: Matplotlib, Seaborn
- **Métricas**: scikit-learn
- **Interfaz Gráfica**: Streamlit
- **Notebook**: Jupyter
- **Control de Versiones**: Git

## Autores

<table>
  <tr>
    <td align="center">
      <img src="profiles/linich.jpg" width="150px;" alt="Jorge Soto"/><br />
      <sub><b>Jorge Soto</b></sub><br />
      <a href="https://github.com/Linich14">@Linich14</a>
    </td>
    <td align="center">
      <img src="profiles/dpbascur.png" width="150px;" alt="Daniel Peña"/><br />
      <sub><b>Daniel Peña</b></sub><br />
      <a href="https://github.com/DPBascur">@DPBascur</a>
    </td>
  </tr>
</table>

Proyecto desarrollado para el curso INFO1185 - Inteligencia Artificial

## Licencia

Este proyecto es de uso académico.