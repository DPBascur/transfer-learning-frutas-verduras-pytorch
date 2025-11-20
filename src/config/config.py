"""
Configuración general del proyecto de Transfer Learning.
Define parámetros del modelo, rutas de datos, hiperparámetros y clases.
"""

import os
from pathlib import Path


class Config:
    """Configuración centralizada para el proyecto"""
    
    # Rutas del proyecto
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / 'Datos'
    TRAIN_DIR = DATA_DIR / 'train'
    VAL_DIR = DATA_DIR / 'validation'
    TEST_DIR = DATA_DIR / 'test'
    MODELS_DIR = BASE_DIR / 'saved_models'
    
    # Clases seleccionadas para el proyecto
    SELECTED_CLASSES = ['apple', 'pomegranate', 'mango', 'lemon', 'orange']
    NUM_CLASSES = len(SELECTED_CLASSES)
    
    # Parámetros del modelo
    MODEL_NAME = 'mobilenet_v3_large'
    INPUT_SIZE = (224, 224)
    PRETRAINED = True
    
    # Hiperparámetros de entrenamiento
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # Early Stopping
    PATIENCE = 10
    MIN_DELTA = 0.001
    
    # Configuración de dispositivo
    DEVICE = 'cuda'  # 'cuda' o 'cpu'
    NUM_WORKERS = 4
    PIN_MEMORY = True
    
    # Data Augmentation
    AUGMENTATION = {
        'rotation': 30,
        'horizontal_flip': 0.5,
        'vertical_flip': 0.2,
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.2,
    }
    
    # Normalización (ImageNet stats)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    
    # Variantes del modelo
    class ModelVariants:
        """Configuraciones para las dos variantes del modelo"""
        
        # Versión 1: Clasificador simple
        VARIANT_1 = {
            'name': 'simple_classifier',
            'use_batch_norm': False,
            'use_dropout': False,
            'dropout_prob': 0.0,
            'hidden_layers': [],
            'activation': 'relu'
        }
        
        # Versión 2: Clasificador extendido tipo embudo
        VARIANT_2 = {
            'name': 'extended_classifier',
            'use_batch_norm': True,
            'use_dropout': True,
            'dropout_prob_range': (0.2, 0.5),
            'hidden_layers': [512, 256, 128],
            'activation': 'relu'
        }
    
    @classmethod
    def create_dirs(cls):
        """Crea los directorios necesarios para el proyecto"""
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_class_to_idx(cls):
        """Retorna el mapeo de clases a índices"""
        return {class_name: idx for idx, class_name in enumerate(cls.SELECTED_CLASSES)}
    
    @classmethod
    def get_idx_to_class(cls):
        """Retorna el mapeo de índices a clases"""
        return {idx: class_name for idx, class_name in enumerate(cls.SELECTED_CLASSES)}
