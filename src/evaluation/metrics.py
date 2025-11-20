"""
Funciones para cálculo de métricas y generación de visualizaciones.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
from sklearn.metrics import confusion_matrix

from src.config import Config


def calculate_metrics(labels: np.ndarray, 
                     predictions: np.ndarray,
                     probabilities: np.ndarray = None) -> Dict:
    """
    Calcula métricas de clasificación.
    
    Args:
        labels: Etiquetas verdaderas
        predictions: Predicciones del modelo
        probabilities: Probabilidades predichas (opcional)
        
    Returns:
        Diccionario con métricas
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_confusion_matrix(conf_matrix: np.ndarray, 
                         class_names: List[str] = None,
                         normalize: bool = False,
                         figsize: tuple = (10, 8),
                         save_path: str = None):
    """
    Genera y visualiza la matriz de confusión.
    
    Args:
        conf_matrix: Matriz de confusión
        class_names: Nombres de las clases
        normalize: Si True, normaliza la matriz
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    if class_names is None:
        class_names = Config.SELECTED_CLASSES
    
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Matriz de Confusión Normalizada'
    else:
        fmt = 'd'
        title = 'Matriz de Confusión'
    
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proporción' if normalize else 'Cantidad'})
    
    plt.title(title, fontsize=14, pad=20)
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def plot_training_history(history: Dict,
                         figsize: tuple = (15, 5),
                         save_path: str = None):
    """
    Visualiza las curvas de aprendizaje (loss y accuracy).
    
    Args:
        history: Diccionario con historial de entrenamiento
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Curva de Pérdida', fontsize=12, pad=10)
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_title('Curva de Accuracy', fontsize=12, pad=10)
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Accuracy (%)', fontsize=10)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[2].set_title('Learning Rate', fontsize=12, pad=10)
    axes[2].set_xlabel('Epoch', fontsize=10)
    axes[2].set_ylabel('Learning Rate', fontsize=10)
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curvas de aprendizaje guardadas en: {save_path}")
    
    plt.show()


def plot_class_distribution(class_counts: Dict[int, int],
                           class_names: List[str] = None,
                           figsize: tuple = (10, 6),
                           save_path: str = None):
    """
    Visualiza la distribución de clases en el dataset.
    
    Args:
        class_counts: Diccionario con conteo por clase
        class_names: Nombres de las clases
        figsize: Tamaño de la figura
        save_path: Ruta para guardar la figura (opcional)
    """
    if class_names is None:
        class_names = Config.SELECTED_CLASSES
    
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(class_names, counts, color='steelblue', alpha=0.8)
    
    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    plt.title('Distribución de Clases', fontsize=14, pad=20)
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Cantidad de Muestras', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribución de clases guardada en: {save_path}")
    
    plt.show()
