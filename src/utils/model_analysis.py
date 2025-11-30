"""
Utilidades adicionales para análisis y visualización de modelos.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from PIL import Image

from src.config import Config


def count_parameters(model: nn.Module) -> Tuple[int, int, int]:
    """
    Cuenta parámetros del modelo.
    
    Args:
        model: Modelo PyTorch
        
    Returns:
        Tupla (total, entrenables, congelados)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    return total, trainable, frozen


def print_model_summary(model: nn.Module, model_name: str = "Modelo"):
    """
    Imprime resumen detallado del modelo.
    
    Args:
        model: Modelo PyTorch
        model_name: Nombre del modelo
    """
    print("\n" + "=" * 80)
    print(f"RESUMEN DEL {model_name.upper()}")
    print("=" * 80)
    
    total, trainable, frozen = count_parameters(model)
    
    print(f"\nParámetros totales: {total:,}")
    print(f"Parámetros entrenables: {trainable:,} ({100*trainable/total:.2f}%)")
    print(f"Parámetros congelados: {frozen:,} ({100*frozen/total:.2f}%)")
    
    print(f"\nArquitectura del clasificador:")
    print(model.mobilenet.classifier)
    
    print("\n" + "=" * 80)


def visualize_model_architecture(model: nn.Module, save_path: str = None):
    """
    Visualiza la arquitectura del modelo de forma gráfica.
    
    Args:
        model: Modelo PyTorch
        save_path: Ruta para guardar la imagen (opcional)
    """
    try:
        from torchviz import make_dot
        
        # Crear input dummy
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        
        # Generar visualización
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        if save_path:
            dot.render(save_path, format='png', cleanup=True)
            print(f"Arquitectura guardada en: {save_path}.png")
        
        return dot
        
    except ImportError:
        print("Para visualizar la arquitectura, instala: pip install torchviz graphviz")
        return None


def analyze_predictions_errors(model: nn.Module, 
                               data_loader,
                               device: str = 'cuda',
                               num_samples: int = 10):
    """
    Analiza los errores de predicción del modelo.
    
    Args:
        model: Modelo a evaluar
        data_loader: DataLoader con datos
        device: Dispositivo
        num_samples: Número de errores a mostrar
        
    Returns:
        Lista de (imagen, label_real, label_pred, probabilidad)
    """
    model.eval()
    model.to(device)
    
    errors = []
    class_names = Config.SELECTED_CLASSES
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Encontrar errores
            incorrect = predicted != labels
            
            for i in range(len(images)):
                if incorrect[i] and len(errors) < num_samples:
                    errors.append({
                        'image': images[i].cpu(),
                        'true_label': class_names[labels[i].item()],
                        'pred_label': class_names[predicted[i].item()],
                        'probability': probs[i][predicted[i]].item(),
                        'true_prob': probs[i][labels[i]].item()
                    })
            
            if len(errors) >= num_samples:
                break
    
    return errors


def plot_prediction_errors(errors: List[dict], figsize: tuple = (15, 12)):
    """
    Visualiza errores de predicción.
    
    Args:
        errors: Lista de diccionarios con información de errores
        figsize: Tamaño de la figura
    """
    from src.utils.visualization import denormalize_image
    
    num_errors = len(errors)
    cols = 3
    rows = (num_errors + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel() if num_errors > 1 else [axes]
    
    for i, error in enumerate(errors):
        if i >= len(axes):
            break
            
        img = denormalize_image(error['image'])
        
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(
            f"Real: {error['true_label']}\n"
            f"Pred: {error['pred_label']} ({error['probability']:.2%})\n"
            f"True prob: {error['true_prob']:.2%}",
            color='red',
            fontsize=10
        )
    
    # Ocultar ejes vacíos
    for i in range(num_errors, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def get_layer_outputs(model: nn.Module, input_tensor: torch.Tensor,
                      layer_name: str = None) -> dict:
    """
    Extrae las salidas de capas intermedias del modelo.
    
    Args:
        model: Modelo PyTorch
        input_tensor: Tensor de entrada
        layer_name: Nombre de capa específica (opcional)
        
    Returns:
        Diccionario con salidas de capas
    """
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Registrar hooks
    hooks = []
    for name, layer in model.named_modules():
        if layer_name is None or name == layer_name:
            hooks.append(layer.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(input_tensor)
    
    # Remover hooks
    for hook in hooks:
        hook.remove()
    
    return activations


def plot_feature_maps(feature_maps: torch.Tensor, 
                     num_maps: int = 16,
                     figsize: tuple = (15, 10)):
    """
    Visualiza mapas de características.
    
    Args:
        feature_maps: Tensor de mapas [batch, channels, height, width]
        num_maps: Número de mapas a mostrar
        figsize: Tamaño de la figura
    """
    if len(feature_maps.shape) == 4:
        feature_maps = feature_maps[0]  # Tomar primer elemento del batch
    
    num_channels = min(num_maps, feature_maps.shape[0])
    cols = 4
    rows = (num_channels + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.ravel() if num_channels > 1 else [axes]
    
    for i in range(num_channels):
        fmap = feature_maps[i].cpu().numpy()
        axes[i].imshow(fmap, cmap='viridis')
        axes[i].axis('off')
        axes[i].set_title(f'Channel {i}')
    
    # Ocultar ejes vacíos
    for i in range(num_channels, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def create_performance_comparison_plot(results_v1: dict, 
                                       results_v2: dict,
                                       save_path: str = None):
    """
    Crea gráfico comparativo de desempeño entre variantes.
    
    Args:
        results_v1: Resultados de Versión 1
        results_v2: Resultados de Versión 2
        save_path: Ruta para guardar (opcional)
    """
    metrics = ['accuracy', 'precision_avg', 'recall_avg', 'f1_avg']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    v1_values = [results_v1[m] for m in metrics]
    v2_values = [results_v2[m] for m in metrics]
    
    x = np.arange(len(metric_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, v1_values, width, label='V1 (Simple)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, v2_values, width, label='V2 (Extendido)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Métricas', fontsize=12)
    ax.set_ylabel('Valor', fontsize=12)
    ax.set_title('Comparación de Desempeño entre Variantes', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Añadir valores sobre las barras
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico guardado en: {save_path}")
    
    plt.show()
