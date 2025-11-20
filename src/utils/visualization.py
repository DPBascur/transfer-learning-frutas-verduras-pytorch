"""
Utilidades para visualización de imágenes y predicciones.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import List, Tuple

from src.config import Config


def denormalize_image(image: torch.Tensor, mean: List[float] = None, std: List[float] = None) -> np.ndarray:
    """
    Desnormaliza una imagen para visualización.
    
    Args:
        image: Tensor de imagen normalizada [C, H, W]
        mean: Media usada en normalización
        std: Desviación estándar usada en normalización
        
    Returns:
        Imagen desnormalizada como numpy array [H, W, C]
    """
    if mean is None:
        mean = Config.MEAN
    if std is None:
        std = Config.STD
    
    # Clonar para no modificar el original
    img = image.clone()
    
    # Desnormalizar
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    
    # Convertir a numpy y transponer
    img = img.numpy().transpose((1, 2, 0))
    
    # Clip a rango válido
    img = np.clip(img, 0, 1)
    
    return img


def visualize_predictions(images: torch.Tensor,
                         labels: torch.Tensor,
                         predictions: torch.Tensor,
                         probabilities: torch.Tensor,
                         num_samples: int = 8,
                         figsize: tuple = (15, 8),
                         save_path: str = None):
    """
    Visualiza imágenes con sus predicciones y etiquetas verdaderas.
    
    Args:
        images: Batch de imágenes
        labels: Etiquetas verdaderas
        predictions: Predicciones del modelo
        probabilities: Probabilidades de predicción
        num_samples: Número de muestras a visualizar
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    class_names = Config.SELECTED_CLASSES
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(2, num_samples // 2, figsize=figsize)
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Desnormalizar imagen
        img = denormalize_image(images[i].cpu())
        
        # Obtener etiquetas
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predictions[i].item()]
        prob = probabilities[i][predictions[i]].item()
        
        # Determinar color (verde si correcto, rojo si incorrecto)
        color = 'green' if true_label == pred_label else 'red'
        
        # Mostrar imagen
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Real: {true_label}\nPred: {pred_label} ({prob:.2%})',
                         color=color, fontsize=10, pad=5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


def plot_sample_images(data_loader, num_samples: int = 12, figsize: tuple = (15, 10)):
    """
    Muestra muestras aleatorias del dataset.
    
    Args:
        data_loader: DataLoader con imágenes
        num_samples: Número de muestras a mostrar
        figsize: Tamaño de la figura
    """
    class_names = Config.SELECTED_CLASSES
    
    # Obtener un batch
    images, labels = next(iter(data_loader))
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(3, num_samples // 3, figsize=figsize)
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Desnormalizar imagen
        img = denormalize_image(images[i])
        
        # Mostrar imagen
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'{class_names[labels[i].item()]}', 
                         fontsize=10, pad=5)
    
    plt.tight_layout()
    plt.show()
