"""
Módulo principal del proyecto de Transfer Learning.
"""

from .config import Config
from .data import get_data_loaders, get_transforms
from .models import create_model
from .training import Trainer, EarlyStopping
from .evaluation import evaluate_model

__all__ = [
    'Config',
    'get_data_loaders',
    'get_transforms',
    'create_model',
    'Trainer',
    'EarlyStopping',
    'evaluate_model'
]
